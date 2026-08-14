"""
Model Runner module.
Handles loading, inference generation, and VRAM memory lifecycle for LLM judges.
Provides dual architecture runners: CausalLMRunner and Seq2SeqRunner (for mT0-XL),
along with an isolated MockModelRunner for software verification.
"""

import abc
import gc
import json
import os
import re
from typing import Optional, Dict, Any, List
from src.config import ModelEntry
from src.logging_utils import logger


class BaseModelRunner(abc.ABC):
    """Abstract base class for LLM runners."""

    def __init__(self, model_entry: ModelEntry):
        self.model_entry = model_entry
        self.is_mock: bool = False
        self.measured_peak_vram_gb: float = 0.0

    @abc.abstractmethod
    def load(self) -> None:
        """Loads model and tokenizer into GPU memory."""
        pass

    @abc.abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generates raw text response for a given prompt."""
        pass

    @abc.abstractmethod
    def unload(self) -> None:
        """Releases model weights and frees GPU memory."""
        pass


class CausalLMRunner(BaseModelRunner):
    """
    Transformers Causal Language Model runner.
    Used for Llama 3.1, Qwen3, Qwen2.5, Gemma 3 (4B/12B), Aya Expanse, Command R7B, BLOOMZ.
    Supports 4-bit quantization, chat templates, and disabling thinking mode for Qwen3.
    """

    def __init__(self, model_entry: ModelEntry):
        super().__init__(model_entry)
        self.tokenizer = None
        self.model = None
        self._is_loaded = False

    def load(self) -> None:
        """Loads causal model and tokenizer into memory."""
        if self._is_loaded:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        logger.info(f"Loading Causal LM: {self.model_entry.hf_id} (quant={self.model_entry.quantization})")
        hf_id = self.model_entry.hf_id

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_id,
            revision=self.model_entry.revision,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.bos_token

        # Compute dtype & quantization configuration
        is_cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if is_cuda and torch.cuda.is_bf16_supported() else torch.float16

        load_kwargs: Dict[str, Any] = {
            "revision": self.model_entry.revision,
            "trust_remote_code": True,
        }

        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
            load_kwargs["device_map"] = "auto"
            if self.model_entry.quantization == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True
                )
                load_kwargs["quantization_config"] = bnb_config
            else:
                load_kwargs["torch_dtype"] = dtype
        else:
            logger.warning(f"CUDA is not available. Loading {hf_id} on CPU (inference will be slow).")
            load_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
        self._is_loaded = True

        if is_cuda:
            peak_bytes = torch.cuda.max_memory_allocated()
            self.measured_peak_vram_gb = round(peak_bytes / (1024 ** 3), 2)
            logger.info(f"Causal model {self.model_entry.id} loaded. Peak VRAM: {self.measured_peak_vram_gb} GB")
        else:
            logger.info(f"Causal model {self.model_entry.id} loaded on CPU.")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self._is_loaded:
            self.load()

        import torch

        # Format prompt with chat template if configured
        formatted_prompt = prompt
        if self.model_entry.chat_template and self.model_entry.chat_template != "plain_prompt" and hasattr(self.tokenizer, "apply_chat_template"):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            try:
                # Handle Qwen3 thinking mode disabling if supported by tokenizer
                template_kwargs = {}
                if getattr(self.model_entry, "disable_thinking", False):
                    template_kwargs["enable_thinking"] = False

                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs
                )
            except Exception as e:
                logger.debug(f"Chat template application fallback for {self.model_entry.id}: {e}")
                formatted_prompt = f"System: {system_prompt}\n\nUser: {prompt}\nAssistant:" if system_prompt else prompt
        elif system_prompt:
            formatted_prompt = f"{system_prompt}\n\n{prompt}"

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        if torch.cuda.is_available() and hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.model_entry.max_new_tokens,
            "temperature": 0.0,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Clean thinking tags if model emitted any (ensures CoT traces are excluded from judgment extraction)
        if "<think>" in response_text and "</think>" in response_text:
            response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

        if torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated()
            self.measured_peak_vram_gb = max(self.measured_peak_vram_gb, round(peak_bytes / (1024 ** 3), 2))

        return response_text

    def unload(self) -> None:
        """Frees causal model from GPU and runs garbage collection."""
        if not self._is_loaded:
            return

        import torch
        logger.info(f"Unloading causal model {self.model_entry.id} and clearing GPU memory...")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self._is_loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class Seq2SeqRunner(BaseModelRunner):
    """
    Transformers Sequence-to-Sequence (Encoder-Decoder) runner.
    Dedicated architecture for mT0-XL (`bigscience/mt0-xl`).
    Uses AutoModelForSeq2SeqLM and formats prompt directly into the encoder.
    """

    def __init__(self, model_entry: ModelEntry):
        super().__init__(model_entry)
        self.tokenizer = None
        self.model = None
        self._is_loaded = False

    def load(self) -> None:
        """Loads sequence-to-sequence model and tokenizer into memory."""
        if self._is_loaded:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

        logger.info(f"Loading Seq2Seq LM: {self.model_entry.hf_id} (quant={self.model_entry.quantization})")
        hf_id = self.model_entry.hf_id

        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_id,
            revision=self.model_entry.revision,
            trust_remote_code=True
        )

        is_cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if is_cuda and torch.cuda.is_bf16_supported() else torch.float16

        load_kwargs: Dict[str, Any] = {
            "revision": self.model_entry.revision,
            "trust_remote_code": True,
        }

        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
            load_kwargs["device_map"] = "auto"
            if self.model_entry.quantization == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True
                )
                load_kwargs["quantization_config"] = bnb_config
            else:
                load_kwargs["torch_dtype"] = dtype
        else:
            logger.warning(f"CUDA is not available. Loading {hf_id} on CPU.")
            load_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForSeq2SeqLM.from_pretrained(hf_id, **load_kwargs)
        self._is_loaded = True

        if is_cuda:
            peak_bytes = torch.cuda.max_memory_allocated()
            self.measured_peak_vram_gb = round(peak_bytes / (1024 ** 3), 2)
            logger.info(f"Seq2Seq model {self.model_entry.id} loaded. Peak VRAM: {self.measured_peak_vram_gb} GB")
        else:
            logger.info(f"Seq2Seq model {self.model_entry.id} loaded on CPU.")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self._is_loaded:
            self.load()

        import torch

        # Combine system prompt and instruction for seq2seq encoder input
        full_input = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        inputs = self.tokenizer(full_input, return_tensors="pt", truncation=True, max_length=1024)
        if torch.cuda.is_available() and hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.model_entry.max_new_tokens,
            "temperature": 0.0,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        if torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated()
            self.measured_peak_vram_gb = max(self.measured_peak_vram_gb, round(peak_bytes / (1024 ** 3), 2))

        return response_text

    def unload(self) -> None:
        """Frees seq2seq model from memory."""
        if not self._is_loaded:
            return

        import torch
        logger.info(f"Unloading seq2seq model {self.model_entry.id} and clearing GPU memory...")
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self._is_loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class MockModelRunner(BaseModelRunner):
    """
    Test-only mock runner for unit testing and software pilots.
    IMPORTANT: Mock outputs must never enter the empirical results directory.
    """

    def __init__(self, model_entry: ModelEntry):
        super().__init__(model_entry)
        self.is_mock = True
        self.measured_peak_vram_gb = 0.05
        logger.warning(
            f"MockModelRunner instantiated for {model_entry.id}. "
            "Mock outputs must be restricted to software tests only."
        )

    def load(self) -> None:
        pass

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Check if language control question
        if "question:" in prompt.lower() or "language comprehension" in prompt.lower():
            if "capital of france" in prompt.lower() or "parís" in prompt.lower() or "पेरिस" in prompt.lower():
                return json.dumps({"answer": "Paris"})
            return json.dumps({"answer": "Correct"})

        # Deterministic allocation based on condition
        is_identifiable = "identifiable" in prompt.lower() or "maya" in prompt.lower() or "rohit" in prompt.lower() or "elena" in prompt.lower()
        allocation = 60.0 if is_identifiable else 45.0
        return json.dumps({"allocation": allocation})

    def unload(self) -> None:
        gc.collect()


# Alias for backward compatibility if imported elsewhere
HuggingFaceModelRunner = CausalLMRunner


def get_model_runner(model_entry: ModelEntry, use_mock: bool = False) -> BaseModelRunner:
    """
    Factory creating the appropriate runner based on architecture and execution mode.
    Routes 'seq2seq' to Seq2SeqRunner (mT0-XL) and 'causal_lm' to CausalLMRunner.
    """
    if use_mock:
        return MockModelRunner(model_entry)

    arch = getattr(model_entry, "architecture", "causal_lm").lower()
    if arch == "seq2seq":
        return Seq2SeqRunner(model_entry)
    return CausalLMRunner(model_entry)
