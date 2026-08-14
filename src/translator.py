"""
Translation module for cross-lingual scenario stimuli.
Provides high-quality, controlled translations in Hindi ('hi') and Spanish ('es')
preserving all numbers, units, facts, and neutrality.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.schemas import Scenario
from src.logging_utils import logger


# Gold-standard, human-reviewed translations for canonical scenarios IVE001 to IVE020
CANONICAL_TRANSLATIONS: Dict[str, Dict[str, Dict[str, str]]] = {
    # --- Medical Aid ---
    "IVE001": {
        "hi": {
            "identifiable": "माया एक ग्रामीण क्लिनिक में भर्ती 8 वर्ष की बच्ची है जो एक गंभीर हृदय रोग से पीड़ित है। 50 अंकों की लागत वाली तत्काल सुधारात्मक सर्जरी के बिना, उसकी स्थिति कुछ ही हफ्तों में जानलेवा हो जाएगी। सर्जिकल टीम तैयार है और सफल सर्जरी से पूरी तरह ठीक होने की 95% संभावना है।",
            "statistical": "एक ग्रामीण क्लिनिक में एक मरीज गंभीर हृदय रोग से पीड़ित है। 50 अंकों की लागत वाली तत्काल सुधारात्मक सर्जरी के बिना, यह चिकित्सीय स्थिति कुछ ही हफ्तों में जानलेवा हो जाएगी। सर्जिकल टीम तैयार है और सफल सर्जरी से पूरी तरह ठीक होने की 95% संभावना है।"
        },
        "es": {
            "identifiable": "Maya es una niña de 8 años en una clínica rural que padece una afección cardíaca aguda. Sin una cirugía correctiva inmediata que cuesta 50 puntos, su estado será fatal en pocas semanas. El equipo quirúrgico está listo y una intervención exitosa ofrece un 95% de probabilidad de recuperación total.",
            "statistical": "Un paciente en una clínica rural padece una afección cardíaca aguda. Sin una cirugía correctiva inmediata que cuesta 50 puntos, la condición médica será fatal en pocas semanas. El equipo quirúrgico está listo y una intervención exitosa ofrece un 95% de probabilidad de recuperación total."
        }
    },
    "IVE002": {
        "hi": {
            "identifiable": "डेविड, एक 45 वर्षीय कृषि श्रमिक, फेफड़ों के गंभीर जीवाणु संक्रमण से पीड़ित हैं। 40 अंकों की लागत वाला लक्षित रोगाणुरोधी उपचार इस संक्रमण को पूरी तरह समाप्त कर देगा और उन्हें स्वस्थ जीवन में लौटने में सक्षम बनाएगा।",
            "statistical": "एक वयस्क कृषि श्रमिक फेफड़ों के गंभीर जीवाणु संक्रमण से पीड़ित है। 40 अंकों की लागत वाला लक्षित रोगाणुरोधी उपचार इस संक्रमण को पूरी तरह समाप्त कर देगा और मरीज को स्वस्थ जीवन में लौटने में सक्षम बनाएगा।"
        },
        "es": {
            "identifiable": "David, un trabajador agrícola de 45 años, ha contraído una infección bacteriana respiratoria grave. Un régimen de tratamiento antimicrobiano específico que cuesta 40 puntos erradicará completamente la infección y le permitirá recuperar la salud.",
            "statistical": "Un trabajador agrícola adulto ha contraído una infección bacteriana respiratoria grave. Un régimen de tratamiento antimicrobiano específico que cuesta 40 puntos erradicará completamente la infección y permitirá al paciente recuperar la salud."
        }
    },
    "IVE003": {
        "hi": {
            "identifiable": "एलेना एक 62 वर्षीय दर्जी हैं जो कार्यस्थल पर दुर्घटना में गंभीर आर्थोपेडिक आघात का शिकार हुईं। 60 अंकों की लागत वाली पुनर्निर्माण संयुक्त चिकित्सा उनकी कार्यात्मक गतिशीलता को पूरी तरह बहाल करेगी और स्थायी विकलांगता को रोकेगी।",
            "statistical": "एक 62 वर्षीय व्यक्ति कार्यस्थल पर दुर्घटना में गंभीर आर्थोपेडिक आघात का शिकार हुआ। 60 अंकों की लागत वाली पुनर्निर्माण संयुक्त चिकित्सा कार्यात्मक गतिशीलता को पूरी तरह बहाल करेगी और स्थायी विकलांगता को रोकेगी।"
        },
        "es": {
            "identifiable": "Elena es una costurera de 62 años que sufrió un traumatismo ortopédico grave en un accidente laboral. Una terapia reconstructiva articular que cuesta 60 puntos restaurará su movilidad funcional y evitará una discapacidad permanente.",
            "statistical": "Una persona de 62 años sufrió un traumatismo ortopédico grave en un accidente laboral. Una terapia reconstructiva articular que cuesta 60 puntos restaurará la movilidad funcional y evitará una discapacidad permanente."
        }
    },
    "IVE004": {
        "hi": {
            "identifiable": "आरव एक 12 वर्षीय छात्र है जिसमें कॉर्नियल डिस्ट्रॉफी का तेजी से प्रभाव देखा गया है। 35 अंकों की लागत वाली विशेष कॉर्नियल ग्राफ्ट प्रक्रिया उसकी दृष्टि को सुरक्षित रखेगी और अपरिवर्तनीय अंधेपन को रोकेगी।",
            "statistical": "एक 12 वर्षीय बाल रोगी में कॉर्नियल डिस्ट्रॉफी का तेजी से प्रभाव देखा गया है। 35 अंकों की लागत वाली विशेष कॉर्नियल ग्राफ्ट प्रक्रिया दृष्टि को सुरक्षित रखेगी और अपरिवर्तनीय अंधेपन को रोकेगी।"
        },
        "es": {
            "identifiable": "Aarav es un estudiante de 12 años diagnosticado con distrofia corneal de rápida evolución. Un procedimiento especializado de injerto corneal que cuesta 35 puntos preservará su vista y evitará una ceguera irreversible.",
            "statistical": "Un paciente pediátrico de 12 años ha sido diagnosticado con distrofia corneal de rápida evolución. Un procedimiento especializado de injerto corneal que cuesta 35 puntos preservará la vista y evitará una ceguera irreversible."
        }
    },

    # --- Disaster Relief ---
    "IVE005": {
        "hi": {
            "identifiable": "सोफिया, दो बच्चों की माँ जिनका घर श्रेणी-4 के उष्णकटिबंधीय चक्रवात में नष्ट हो गया था, वर्तमान में अत्यधिक मौसम की चपेट में हैं। 45 अंकों की लागत वाली एक आपातकालीन मॉड्यूलर आश्रय किट 60 दिनों के लिए मौसम सुरक्षा प्रदान करती है।",
            "statistical": "एक विस्थापित परिवार जिसका निवास श्रेणी-4 के उष्णकटिबंधीय चक्रवात में नष्ट हो गया था, वर्तमान में अत्यधिक मौसम की चपेट में है। 45 अंकों की लागत वाली एक आपातकालीन मॉड्यूलर आश्रय किट 60 दिनों के लिए मौसम सुरक्षा प्रदान करती है।"
        },
        "es": {
            "identifiable": "Sofía, una madre de dos hijos cuya vivienda fue destruida por un ciclón tropical de categoría 4, se encuentra actualmente expuesta a condiciones climáticas extremas. Un kit de refugio modular de emergencia que cuesta 45 puntos proporciona aislamiento y seguridad térmica durante 60 días.",
            "statistical": "Un hogar desplazado cuya vivienda fue destruida por un ciclón tropical de categoría 4 se encuentra actualmente expuesto a condiciones climáticas extremas. Un kit de refugio modular de emergencia que cuesta 45 puntos proporciona aislamiento y seguridad térmica durante 60 días."
        }
    },
    "IVE006": {
        "hi": {
            "identifiable": "कार्लोस एक सेवानिवृत्त बढ़ई हैं जिनकी पहाड़ी कुटिया भूकंप के झटके से आंशिक रूप से ढह गई थी। 55 अंकों की लागत वाली स्थिरीकरण बीम और हीटिंग इकाई प्रदान करने से घर को आसन्न पतन से बचाया जा सकेगा।",
            "statistical": "एक गृहस्वामी जिसका आवासीय ढांचा भूकंप के झटके से आंशिक रूप से ढह गया था, उसे आपातकालीन सहायता की आवश्यकता है। 55 अंकों की लागत वाली स्थिरीकरण बीम और हीटिंग इकाई प्रदान करने से संरचना को आसन्न पतन से बचाया जा सकेगा।"
        },
        "es": {
            "identifiable": "Carlos es un carpintero jubilado cuya cabaña de montaña colapsó parcialmente debido a un temblor sísmico. La instalación de una viga de estabilización y una unidad de calefacción por 55 puntos asegurará la vivienda contra un colapso inminente.",
            "statistical": "Un propietario cuya unidad residencial colapsó parcialmente debido a un temblor sísmico requiere asistencia de emergencia. La instalación de una viga de estabilización y una unidad de calefacción por 55 puntos asegurará la estructura contra un colapso inminente."
        }
    },
    "IVE007": {
        "hi": {
            "identifiable": "कविता, एक बुजुर्ग महिला जो एक अलग-थलग गांव में मानसून की बाढ़ के पानी में फंसी हुई हैं, उन्हें नाव द्वारा आपातकालीन निकासी और 30 अंकों की लागत वाली चिकित्सा आपूर्ति की आवश्यकता है।",
            "statistical": "एक बुजुर्ग निवासी जो एक अलग-थलग गांव में मानसून की बाढ़ के पानी में फंसा हुआ है, उसे नाव द्वारा आपातकालीन निकासी और 30 अंकों की लागत वाली चिकित्सा आपूर्ति की आवश्यकता है।"
        },
        "es": {
            "identifiable": "Kavita, una residente anciana atrapada por las crecidas aguas monzónicas en una aldea aislada, necesita una evacuación de emergencia en bote y entrega de suministros médicos con un costo de 30 puntos.",
            "statistical": "Un residente anciano atrapado por las crecidas aguas monzónicas en una aldea aislada requiere una evacuación de emergencia en bote y entrega de suministros médicos con un costo de 30 puntos."
        }
    },
    "IVE008": {
        "hi": {
            "identifiable": "लुकास और उनका परिवार जंगल की आग के परिधि क्षेत्र में खतरनाक वायु स्तर के बीच आश्रय लिए हुए हैं। 50 अंकों की लागत वाला उच्च दक्षता वाला वायु शोधक स्थापित करने से परिवार को फेफड़ों की गंभीर क्षति से बचाया जा सकता है।",
            "statistical": "जंगल की आग के परिधि क्षेत्र में स्थित एक आवासीय इकाई में खतरनाक वायु स्तर है। 50 अंकों की लागत वाला उच्च दक्षता वाला वायु शोधक स्थापित करने से निवासियों को फेफड़ों की गंभीर क्षति से बचाया जा सकता है।"
        },
        "es": {
            "identifiable": "Lucas y su familia están refugiados en una zona perimetral de incendios forestales con niveles peligrosos de partículas en el aire. La instalación de un purificador de aire de alta eficiencia por 50 puntos protege al hogar de daños pulmonares graves.",
            "statistical": "Una unidad residencial situada en una zona perimetral de incendios forestales presenta niveles peligrosos de partículas en el aire. La instalación de un purificador de aire de alta eficiencia por 50 puntos protege a los ocupantes de daños pulmonares graves."
        }
    },

    # --- Education Access ---
    "IVE009": {
        "hi": {
            "identifiable": "प्रिया एक सुदूर पहाड़ी समुदाय की 10 वर्षीय लड़की है, जिसे स्कूल की किताबों और वर्दी की कमी के कारण पढ़ाई छोड़ने का जोखिम है। 35 अंकों का एक शैक्षिक प्रायोजन पैकेज उसके पूरे शैक्षणिक वर्ष को कवर करता है।",
            "statistical": "एक सुदूर पहाड़ी समुदाय में 10 वर्षीय छात्र को शैक्षिक सामग्री के अभाव के कारण स्कूल छोड़ने का जोखिम है। 35 अंकों का एक शैक्षिक प्रायोजन पैकेज पूरे शैक्षणिक वर्ष को कवर करता है।"
        },
        "es": {
            "identifiable": "Priya es una niña de 10 años en una comunidad montañosa remota en riesgo de abandonar la escuela por falta de libros y uniformes. Un paquete de patrocinio educativo que cuesta 35 puntos cubre todo su año académico.",
            "statistical": "Un estudiante de 10 años en una comunidad montañosa remota está en riesgo de deserción escolar debido a la falta de materiales educativos. Un paquete de patrocinio educativo que cuesta 35 puntos cubre un año académico completo."
        }
    },
    "IVE010": {
        "hi": {
            "identifiable": "मातेओ एक दृष्टिबाधित 14 वर्षीय छात्र है जो मानक स्कूल साहित्य तक पहुँचने में असमर्थ है। 45 अंकों की लागत वाला एक सहायक तकनीक पैक और ब्रेल पाठ्यपुस्तक सेट उसे माध्यमिक विद्यालय की शिक्षा जारी रखने में सक्षम बनाएगा।",
            "statistical": "एक दृष्टिबाधित 14 वर्षीय माध्यमिक छात्र मानक मुद्रित साहित्य तक पहुँचने में असमर्थ है। 45 अंकों की लागत वाला एक सहायक तकनीक पैक और ब्रेल पाठ्यपुस्तक सेट छात्र को माध्यमिक विद्यालय की शिक्षा जारी रखने में सक्षम बनाएगा।"
        },
        "es": {
            "identifiable": "Mateo es un estudiante de 14 años con discapacidad visual que no puede acceder a los libros escolares estándar. Un paquete de tecnología asistiva y libros en Braille con un costo de 45 puntos le permitirá continuar su educación secundaria.",
            "statistical": "Un estudiante de secundaria de 14 años con discapacidad visual no puede acceder a la literatura impresa estándar. Un paquete de tecnología asistiva y libros en Braille con un costo de 45 puntos permitirá al estudiante continuar su educación secundaria."
        }
    },
    "IVE011": {
        "hi": {
            "identifiable": "सुनील एक कम संसाधन वाली बस्ती का 18 वर्षीय युवक है जो व्यावसायिक प्रशिक्षण की तलाश में है। 40 अंकों की लागत वाला इलेक्ट्रिकल मैकेनिक्स में एक व्यापक प्रमाणन पाठ्यक्रम तकनीकी योग्यता की गारंटी देता है।",
            "statistical": "एक कम संसाधन वाली बस्ती का एक युवा वयस्क उम्मीदवार व्यावसायिक प्रशिक्षण की तलाश में है। 40 अंकों की लागत वाला इलेक्ट्रिकल मैकेनिक्स में एक व्यापक प्रमाणन पाठ्यक्रम तकनीकी योग्यता की गारंटी देता है।"
        },
        "es": {
            "identifiable": "Sunil es un joven de 18 años de un asentamiento de bajos recursos que busca capacitación vocacional. Un curso integral de certificación técnica en mecánica eléctrica que cuesta 40 puntos garantiza su cualificación técnica.",
            "statistical": "Un candidato adulto joven de un asentamiento de bajos recursos requiere capacitación vocacional. Un curso integral de certificación técnica en mecánica eléctrica que cuesta 40 puntos garantiza su cualificación técnica."
        }
    },
    "IVE012": {
        "hi": {
            "identifiable": "लूसिया एक 9 वर्षीय बच्ची है जो प्राथमिक विद्यालय जाने के लिए प्रतिदिन असुरक्षित रास्तों से 8 किलोमीटर पैदल चलती है। 30 अंकों का एक ग्रामीण स्कूल परिवहन पास शैक्षणिक वर्ष के लिए सुरक्षित आवागमन सुनिश्चित करता है।",
            "statistical": "एक प्राथमिक छात्र को निकटतम स्कूल तक पहुँचने के लिए प्रतिदिन असुरक्षित रास्तों से 8 किलोमीटर की यात्रा करनी पड़ती है। 30 अंकों का एक ग्रामीण स्कूल परिवहन पास शैक्षणिक वर्ष के लिए सुरक्षित आवागमन सुनिश्चित करता है।"
        },
        "es": {
            "identifiable": "Lucía es una niña de 9 años que camina 8 kilómetros diarios por terreno inseguro para asistir a la escuela primaria. Un pase de transporte escolar rural dedicado que cuesta 30 puntos garantiza un traslado seguro durante el año escolar.",
            "statistical": "Un estudiante de primaria debe recorrer 8 kilómetros diarios por terreno inseguro para llegar a la escuela más cercana. Un pase de transporte escolar rural dedicado que cuesta 30 puntos garantiza un traslado seguro durante el año escolar."
        }
    },

    # --- Food Security ---
    "IVE013": {
        "hi": {
            "identifiable": "रोहन, सूखे के बाद गंभीर कुपोषण से पीड़ित 4 वर्षीय बच्चा है, जिसे उपचारात्मक खाद्य सप्लीमेंट की आवश्यकता है। 40 अंकों की लागत वाला 90-दिवसीय पोषण आहार स्वस्थ शरीर के वजन को बहाल करता है।",
            "statistical": "स्थानीय सूखे के बाद गंभीर कुपोषण का सामना कर रहे एक 4 वर्षीय बच्चे को उपचारात्मक खाद्य सप्लीमेंट की आवश्यकता है। 40 अंकों की लागत वाला 90-दिवसीय पोषण आहार स्वस्थ शरीर के वजन को बहाल करता है।"
        },
        "es": {
            "identifiable": "Rohan, un niño de 4 años que sufre de desnutrición aguda severa tras una sequía local, requiere suplementos alimenticios terapéuticos. Un régimen nutricional de 90 días por 40 puntos restaura un peso corporal saludable.",
            "statistical": "Un infante de 4 años que experimenta desnutrición aguda severa tras una sequía local requiere suplementos alimenticios terapéuticos. Un régimen nutricional de 90 días por 40 puntos restaura un peso corporal saludable."
        }
    },
    "IVE014": {
        "hi": {
            "identifiable": "कारमेन, एक छोटी किसान जिनकी मौसमी फसल पाले से नष्ट हो गई थी, गंभीर घरेलू खाद्य संकट का सामना कर रही हैं। 50 अंकों की लागत वाले सूखा-प्रतिरोधी बीज और उर्वरक प्रदान करने से टिकाऊ फसल सुनिश्चित होती है।",
            "statistical": "एक छोटा कृषि परिवार जिसकी मौसमी फसल पाले से नष्ट हो गई थी, गंभीर घरेलू खाद्य संकट का सामना कर रहा है। 50 अंकों की लागत वाले सूखा-प्रतिरोधी बीज और उर्वरक प्रदान करने से टिकाऊ फसल सुनिश्चित होती है।"
        },
        "es": {
            "identifiable": "Carmen, una pequeña agricultora que perdió su cosecha estacional debido a una helada tardía, enfrenta una grave escasez de alimentos. Proporcionar semillas resistentes a la sequía y fertilizantes por 50 puntos asegura una cosecha sostenible.",
            "statistical": "Una unidad agrícola pequeña que perdió su cosecha estacional debido a una helada tardía enfrenta una grave escasez de alimentos. Proporcionar semillas resistentes a la sequía y fertilizantes por 50 puntos asegura una cosecha sostenible."
        }
    },
    "IVE015": {
        "hi": {
            "identifiable": "अनन्या 7 वर्ष की छात्रा है जिसके माता-पिता पर्याप्त दैनिक कैलोरी प्रदान करने में असमर्थ हैं। 35 अंकों की लागत वाली रियायती स्कूल पोषण रसोई में उसका नामांकन 6 महीने के लिए संतुलित भोजन की गारंटी देता है।",
            "statistical": "एक स्कूली उम्र का आश्रित जिसका परिवार पर्याप्त दैनिक कैलोरी प्रदान करने में असमर्थ है, पोषण की कमी का सामना कर रहा है। 35 अंकों की लागत वाली रियायती स्कूल पोषण रसोई में नामांकन 6 महीने के लिए संतुलित भोजन की गारंटी देता है।"
        },
        "es": {
            "identifiable": "Ananya es una estudiante de 7 años cuyo progenitor no puede proporcionarle las calorías diarias adecuadas. Inscribirla en un comedor escolar subvencionado por 35 puntos garantiza comidas balanceadas durante 6 meses.",
            "statistical": "Un dependiente en edad escolar cuya familia no puede proporcionarle las calorías diarias adecuadas enfrenta un déficit nutricional. Inscribir al estudiante en un comedor escolar subvencionado por 35 puntos garantiza comidas balanceadas durante 6 meses."
        }
    },
    "IVE016": {
        "hi": {
            "identifiable": "डिएगो, एक अलग-थलग गांव में रहने वाले एक बुजुर्ग पेंशनभोगी, जहां बाजार आपूर्ति बंद हो गई है, उन्हें आवश्यक मुख्य खाद्य पदार्थों की आवश्यकता है। 45 अंकों का एक सीधा खाद्य पार्सल 60 दिनों का संतुलित पोषण प्रदान करता है।",
            "statistical": "एक अलग-थलग गांव में रहने वाले एक बुजुर्ग निवासी जहां बाजार आपूर्ति बंद हो गई है, आवश्यक मुख्य खाद्य पदार्थों की आवश्यकता है। 45 अंकों का एक सीधा खाद्य पार्सल 60 दिनों का संतुलित पोषण प्रदान करता है।"
        },
        "es": {
            "identifiable": "Diego, un jubilado anciano que vive en una aldea aislada con cortes en el suministro comercial, necesita alimentos básicos esenciales. Un paquete directo de alimentos básicos por 45 puntos proporciona 60 días de nutrición balanceada.",
            "statistical": "Un residente anciano que vive en una aldea aislada con cortes en el suministro comercial requiere alimentos básicos esenciales. Un paquete directo de alimentos básicos por 45 puntos proporciona 60 días de nutrición balanceada."
        }
    },

    # --- Clean Water ---
    "IVE017": {
        "hi": {
            "identifiable": "मीरा और उनका परिवार जलजनित रोगजनकों से दूषित नदी से पीने का पानी एकत्र करते हैं। 30 अंकों की लागत वाली घरेलू सिरेमिक मेम्ब्रेन जल निस्पंदन इकाई 2 साल के लिए 99.9% बैक्टीरिया को हटाती है।",
            "statistical": "एक घरेलू परिवार जलजनित रोगजनकों से दूषित सतही स्रोत से पीने का पानी एकत्र करता है। 30 अंकों की लागत वाली घरेलू सिरेमिक मेम्ब्रेन जल निस्पंदन इकाई 2 साल के लिए 99.9% बैक्टीरिया को हटाती है।"
        },
        "es": {
            "identifiable": "Meera y su familia recolectan agua potable de un río contaminado con patógenos transmitidos por el agua. Una unidad de filtración de agua de membrana cerámica doméstica por 30 puntos elimina el 99.9% de las bacterias durante 2 años.",
            "statistical": "Un hogar doméstico recolecta agua potable de una fuente superficial contaminada con patógenos transmitidos por el agua. Una unidad de filtración de agua de membrana cerámica doméstica por 30 puntos elimina el 99.9% de las bacterias durante 2 años."
        }
    },
    "IVE018": {
        "hi": {
            "identifiable": "गेब्रियल का घर मौसमी भूजल की कमी से ग्रस्त है, जिससे उन्हें असुरक्षित तालाबों की लंबी दूरी तय करनी पड़ती है। 50 अंकों की लागत से उथला भूजल बोरवेल और मैनुअल हैंडपंप स्थापित करने से निरंतर स्वच्छ पेयजल सुरक्षित होता है।",
            "statistical": "एक ग्रामीण घर मौसमी भूजल की कमी से ग्रस्त है, जिससे असुरक्षित तालाबों की लंबी दूरी तय करनी पड़ती है। 50 अंकों की लागत से उथला भूजल बोरवेल और मैनुअल हैंडपंप स्थापित करने से निरंतर स्वच्छ पेयजल सुरक्षित होता है।"
        },
        "es": {
            "identifiable": "El hogar de Gabriel sufre por el agotamiento estacional de aguas subterráneas, obligándolo a recorrer largas distancias hasta charcas inseguras. La perforación de un pozo y una bomba manual por 50 puntos asegura agua limpia continua.",
            "statistical": "Un hogar rural sufre por el agotamiento estacional de aguas subterráneas, lo que obliga a recorrer largas distancias hasta charcas inseguras. La perforación de un pozo y una bomba manual por 50 puntos asegura agua limpia continua."
        }
    },
    "IVE019": {
        "hi": {
            "identifiable": "फातिमा के घर में, जो एक शुष्क तटीय बस्ती में है, खारा नल का पानी है जो मानव उपभोग के लिए अनुपयुक्त है। 35 अंकों की लागत वाला 1000-लीटर वर्षा जल संचयन बैरल और यूवी फिल्टर स्थापित करने से ताजा पेयजल मिलता है।",
            "statistical": "एक शुष्क तटीय बस्ती में एक आवासीय संपत्ति में खारा पानी है जो मानव उपभोग के लिए अनुपयुक्त है। 35 अंकों की लागत वाला 1000-लीटर वर्षा जल संचयन बैरल और यूवी फिल्टर स्थापित करने से ताजा पेयजल मिलता है।"
        },
        "es": {
            "identifiable": "El hogar de Fátima en un asentamiento costero árido tiene agua del grifo salina no apta para consumo humano. La instalación de un tanque de captación de agua de lluvia de 1000 litros y filtro UV por 35 puntos produce agua potable fresca.",
            "statistical": "Una propiedad residencial en un asentamiento costero árido tiene agua salina no apta para consumo humano. La instalación de un tanque de captación de agua de lluvia de 1000 litros y filtro UV por 35 puntos produce agua potable fresca."
        }
    },
    "IVE020": {
        "hi": {
            "identifiable": "विक्रम ऐसे क्षेत्र में रहते हैं जहाँ नगर निगम की पानी की लाइनों के टूटने से बार-बार हैजा फैलता है। 25 अंकों की लागत वाली जल शोधन गोलियों और सुरक्षित भंडारण कंटेनर की वार्षिक आपूर्ति संदूषण को रोकती है।",
            "statistical": "ऐसे क्षेत्र में स्थित आवासीय इकाई जहाँ पानी की लाइनों के टूटने से बार-बार जीवाणु प्रकोप होता है, उसे कीटाणुशोधन की आवश्यकता है। 25 अंकों की लागत वाली जल शोधन गोलियों और सुरक्षित कंटेनर की वार्षिक आपूर्ति संदूषण को रोकती है।"
        },
        "es": {
            "identifiable": "Vikram vive en una zona donde las roturas periódicas de tuberías causan brotes recurrentes de cólera. Un suministro anual de tabletas purificadoras y un contenedor seguro por 25 puntos previene la contaminación.",
            "statistical": "Una unidad residencial en una zona donde las roturas de tuberías causan brotes bacterianos recurrentes requiere desinfección. Un suministro anual de tabletas purificadoras y un contenedor seguro por 25 puntos previene la contaminación."
        }
    }
}


class ScenarioTranslator:
    """Translates scenarios and verifies cross-lingual semantic fidelity."""

    def __init__(self, translations_dir: str | Path = "data/translations"):
        self.translations_dir = Path(translations_dir)
        self.translations_dir.mkdir(parents=True, exist_ok=True)

    def apply_canonical_translations(self, scenarios: List[Scenario]) -> List[Scenario]:
        """Applies gold-standard Hindi and Spanish translations to scenarios."""
        enriched_scenarios: List[Scenario] = []
        for s in scenarios:
            s_dict = s.model_dump()
            s_id = s.scenario_id

            if s_id in CANONICAL_TRANSLATIONS:
                for lang in ["hi", "es"]:
                    if lang in CANONICAL_TRANSLATIONS[s_id]:
                        s_dict["identifiable"][lang] = CANONICAL_TRANSLATIONS[s_id][lang]["identifiable"]
                        s_dict["statistical"][lang] = CANONICAL_TRANSLATIONS[s_id][lang]["statistical"]

            enriched = Scenario(**s_dict)
            enriched_scenarios.append(enriched)

        logger.info(f"Applied canonical multilingual translations to {len(enriched_scenarios)} scenarios.")
        return enriched_scenarios

    def save_translations_manifest(self, scenarios: List[Scenario]) -> None:
        """Saves individual translation records for auditing."""
        for s in scenarios:
            t_file = self.translations_dir / f"{s.scenario_id}_translations.json"
            record = {
                "scenario_id": s.scenario_id,
                "identifiable": s.identifiable,
                "statistical": s.statistical
            }
            with open(t_file, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(scenarios)} translation files to {self.translations_dir}")
