**![][image1]                                                                                      ![D:\\f6o0om\\شعارات\\صتنورة26.png][image2]**

**Al Imam Mohammad Ibn Saud Islamic University**

**College of Computer and Information Sciences**

**Information Technology Department**

***Qurrā – An AI-Powered Quran Reciter and Verse Identification System***

**By:**

| Name | ID | Department |
| :---- | :---- | :---- |
| *Abdusamii Abdulazeez* | *443022312* | *Information Technology* |
| *Ziyad Abdulaziz Algufari* | *444003779* | *Information Technology* |
|  |  |  |

**Supervised by:** 

*Dr. Mohd Fazil*

Project Submitted in Partial Fulfillment for the Degree of B.Sc. in "Information Technology"

**Fall 2021**

**![A logo of a companyAI-generated content may be incorrect.][image1]                                                                                      ![D:\\f6o0om\\شعارات\\صتنورة26.png][image2]**

**Al Imam Mohammad Ibn Saud Islamic University**

**College of Computer and Information Sciences**

**Information Technology Department**

***Qurrā – An AI-Powered Quran Reciter and Verse Identification System***

**By:**

| Name | ID | Department |
| :---- | :---- | :---- |
| *Abdusamii Abdulazeez* | *443022312* | *Information Technology* |
| *Ziyad Abdulaziz Algufari* | *444003779* | *Information Technology* |

| Supervisor Name |  |
| :---- | :---- |
| **Date**  |   |
| **Signature**  |   |

# **Declaration** {#declaration}

We ***Abdusamii Abdulazeez*** and ***Ziyad Abdulaziz Algufari*** being members of final year project group number **171**, declare that this report contains only work completed by members of our group except for information obtained in a legitimate way from literature, company or university sources. All information from these other sources has been duly referenced and acknowledged in accordance with the University Policy on Plagiarism.

Furthermore, we declare that in completing the project, the individual group members had the following responsibilities and contributed in the following proportions to the final outcomes of the project: 

| Student ID | Responsibilities | Contribution % | Signature |
| ----- | ----- | ----- | ----- |
|  443022312 | Complete React Native mobile application development User interface design and cross-platform implementation  Major contribution to reciter model (Random Forest & BLSTM)  API development and backend integration   Mobile-server communication and optimization  User experience testing and refinement |  50% |   |
| 444003779  | Complete Tarteel Whisper model integration  Ayah identification system development  Arabic speech recognition and fuzzy matching  API development and endpoint implementation  Minor contribution to reciter model development Testing and validation of speech recognition  |  50% |   |

# **Acknowledgment** {#acknowledgment}

First and foremost, we praise and thank Almighty Allah (سبحانه وتعالى) for His countless blessings, guidance, and for granting us the knowledge, patience, and perseverance to complete this project. Without His divine grace, this work would not have been possible.

We extend our heartfelt gratitude to our supervisor Dr. Mohd Fazil for his invaluable guidance, unwavering support, and constructive feedback throughout the development of the Qurrā system.

We would like to acknowledge the Islamic scholarly tradition that inspired this work, particularly the renowned Quran reciters whose beautiful recitations form the foundation of our dataset. Their devotion to preserving the art of Quranic recitation has made this technological advancement possible.

We express our deep appreciation to the maintainers of EveryAyah.com for their invaluable service in preserving and providing access to high-quality Quranic recitations. Their dedication to making authentic recitations freely available has been fundamental to our research and model training.

Special appreciation goes to the developers of Tarteel AI whose Arabic speech recognition model enabled the Ayah identification capabilities, and the contributors to the various machine learning and audio processing libraries that made this project feasible.

We express our sincere gratitude to our families for their unconditional love, prayers, and continuous encouragement throughout our academic journey. Their support and understanding during the challenging phases of this project have been our source of strength.

Finally, we thank our friends and colleagues who provided valuable feedback, testing assistance, and moral support during the development and evaluation phases of the Qurrā system. Their contributions, both technical and emotional, have been deeply appreciated.

May Allah (سبحانه وتعالى) accept this humble effort as a contribution to Islamic educational technology and grant benefit to the Muslim community worldwide.

# **Abstract** {#abstract}

This research presents the development of Qurrā, an AI-driven mobile application for identifying Qur'ān reciters and verses through audio input, functioning as a "Shazam for Qur'ānic recitations". The project addresses a significant gap in Islamic technology by providing an accessible tool for Muslims worldwide to identify both reciters and specific verses (Āyāt) from audio they encounter.

The system employs dual machine learning approaches: Random Forest classification for efficient reciter identification and Bidirectional Long Short-Term Memory (BLSTM) networks for enhanced accuracy. Additionally, Tarteel AI's Whisper model enables precise Ayah identification through Arabic speech recognition and advanced text matching algorithms.

The application features a React Native frontend for cross-platform mobile support, coupled with a robust Python Flask backend for audio processing and model inference. The system processes audio input in real-time, providing confidence scores and detailed information about identified reciters and verses while supporting both recorded and uploaded audio files.

Key achievements include successful implementation of dual recognition capabilities, deployment of multiple ML models achieving 93.57% reciter identification accuracy, and creation of an intuitive mobile interface. The system contributes to technological advancement in Islamic applications and the broader field of Arabic audio processing.

**Keywords:** Qur’ān Reciter Identification, Āyah Recognition, Arabic Speech Processing, Audio Feature Extraction, MFCC, Machine Learning, BLSTM, React Native, Flask, Islamic Educational Technology.

# **Abstract (in Arabic)** {#abstract-(in-arabic)}

يُقَدِّمُ هذا البحث تطوير تطبيق جوّال مُبتَكَر يُدعى "قُرَّاء"، يعتمد على الذكاء الاصطناعي للتعرّف على قارئ القرآن الكريم والآيات المُرتّلة من خلال التّسجيلات الصوتيّة، حيث يعمل كنظام مشابه لتطبيق "Shazam" ولكن لتلاوات القرآن الكريم. يهدف المشروع إلى سدّ فجوة تقنيّة في مجال التكنولوجيا الإسلاميّة من خلال توفير أداة تُمكِّن المسلمين حول العالم من التعرّف بسهولة على القارئ والآية التي يسمعونها.  
يعتمد النظام على نهجين في تعلم الآلة: تصنيف الغابات العشوائية (Random Forest) لتحديد هوية القارئ بسرعة وكفاءة، وشبكات الذاكرة طويلة المدى ثنائية الاتجاه (BLSTM) لتعزيز دقة التعرّف. كما يستفيد من نموذج Whisper المطوّر من قبل Tarteel AI للتعرّف على الآيات باستخدام تقنيات متقدّمة للتعرف على الكلام العربي ومطابقة النصوص.  
يتميّز التطبيق بواجهة أمامية مبنيّة باستخدام React Native لدعم الأنظمة المتعددة (iOS وAndroid)، إلى جانب خادم خلفي قوي باستخدام Python Flask لمعالجة الصوت وتنفيذ نماذج التعرّف. يتيح النظام معالجة التّسجيلات في الوقت الحقيقي، مع عرض معلومات تفصيلية عن القارئ والآية المصاحبة، إضافة إلى تقديم درجة ثقة لكل نتيجة. يدعم التطبيق التّسجيل المباشر أو رفع ملفات صوتيّة مُخزّنة مسبقًا.  
تشمل أبرز الإنجازات تنفيذ دقّة تعرّف بلغت 93.57٪ في تحديد القارئ، وتكامل عدّة نماذج تعلم آلي، وتصميم واجهة مستخدم سهلة وبديهية. يُعد هذا المشروع خطوة نوعية نحو تطوير حلول ذكيّة في التطبيقات الإسلامية، ويُسهم في مجال معالجة الصوت العربي وتقنيات التعليم الديني الرقمي.

**الكلمات المفتاحية:**  
التعرّف على قارئ القرآن، التعرّف على الآيات، معالجة الكلام العربي، استخراج الميزات الصوتية، MFCC، تعلم الآلة، BLSTM، React Native، Flask، التكنولوجيا التعليمية الإسلامية..

# **List of Abbreviations** {#list-of-abbreviations}

| AI | Artificial Intelligence |
| :---- | :---- |
| API | Application Programming Interface |
| Ayat | Verses of the Quran (singular: Āyah) |
| Basmallah | The recitation of "Bismillah-ir-Rahman-ir-Raheem" (In the name of Allah, the Most Gracious, the Most Merciful) |
| BLSTM | Bidirectional Long Short-Term Memory |
| CNN | Convolutional Neural Network |
| CSV | Comma-Seperated Values |
| CV | Cross-Validation |
| Flask | Python web framework for building APIs |
| GPU  | Graphics Processing Unit |
| GRU | Gated Recurrent Unit |
| GTCC | Gammatone Cepstral Coefficients |
| HTTP/HTTPS | Hypertext Transfer Protocol (Secure) |
| IDE | Integrated Development Environment |
| JSON | JavaScript Object Notation |
| JWT | Jason Web Token |
| k-NN | k-Nearest Neighbor |
| LSTM | Long Short-Term Memory |
| MFCC | Mel-Frequency Cepstral Coefficients |
| ML | Machine Learning |
| MVP | Minimum Viable Product |
| npm | Node Package Manager |
| Qirāʾāt | Quranic Recitation Styles |
| Qurrā | Project name meaning "Reciters" in Arabic |
| REST | Representational State Transfer |
| ResNet | Residual Network |
| RNN | Recurrent Neural Network |
| RTL | Right-to-Left (text direction) |
| SNR | Signal-to-Noise Ratio |
| STFT | Short-Time Fourier Transform |
| Surah | Chapter of the Quran |
| SVM | Support Vector Machine |
| Tajweed  | Quranic rules of recitation |
| Tahmeen | Congregational response such as "Ameen" in Surah Al-Fatihah |
| UI/UX | User Interface/User Experience |
| VGG | Visual Geometry Group |
| WAV | Waveform Audio File Format |

**Table of Contents**

**[*Declaration	3*](#declaration)**

[***Acknowledgment	5***](#acknowledgment)

[***Abstract	6***](#abstract)

[***Abstract (in Arabic)	7***](#abstract-\(in-arabic\))

[***List of Abbreviations	8***](#list-of-abbreviations)

[***Table of Contents	10***](#table-of-contents)

[***List of Figures	12***](#list-of-figures)

[***List of Tables	13***](#list-of-tables)

[***1\. Introduction	15***](#introduction)

[1.1. Introduction	15](#introduction-1)

[1.2. Problem Definition	15](#problem-definition)

[1.3. Project Scope	17](#project-scope)

[1.4. Local Impact	18](#local-impact)

[1.5. Global Impact	19](#global-impact)

[1.6. Aims and Objectives	19](#aims-and-objectives)

[1.7. Alternative Solutions	20](#alternative-solutions)

[1.8. Method / Approach	21](#method-/-approach)

[1.9. Project Timeline	23](#project-timeline)

[1.10. Report Structure	26](#report-structure)

[1.11. Summary	26](#summary)

[***2\. Literature Review	29***](#literature-review)

[2.1. Introduction	29](#introduction-2)

[2.2. Background	29](#background)  
[2.2.1.	The Significance of Qur’ānic Audio Processing	29](#the-significance-of-qur’ānic-audio-processing)  
[2.2.2.	The Diversity of Qur’ān Recitation Styles	29](#the-diversity-of-qur’ān-recitation-styles)  
[2.2.3.	Audio Processing in Islamic Applications	30](#audio-processing-in-islamic-applications)  
[2.2.4.	Machine Learning and Deep Learning Approaches	31](#machine-learning-and-deep-learning-approaches)

[2.3. Related Work	31](#related-work)  
[2.3.1.	Traditional Approaches to Reciter Recognition	31](#traditional-approaches-to-reciter-recognition)  
[2.3.2.	Deep Learning Approaches for Reciter Recognition	32](#deep-learning-approaches-for-reciter-recognition)  
[2.3.3.	Arabic Speech Recognition Systems	32](#arabic-speech-recognition-systems)  
[2.3.4.	Hybrid and Multi-Modal Approaches	32](#hybrid-and-multi-modal-approaches)  
[2.3.5.	Challenges and Limitations in Current Systems	33](#challenges-and-limitations-in-current-systems)

[2.4. Summary	34](#summary-1)

[***3\. System Analysis and Design	36***](#system-analysis-and-design)

[3.1. Introduction	36](#introduction-3)

[3.2. System Perspective	36](#system-perspective)

[3.3. Requirements Elicitation Techniques	37](#requirements-elicitation-techniques)

[3.4. System Requirements	37](#system-requirements)  
[3.4.1.	Functional Requirements	37](#functional-requirements)  
[3.4.2.	Non-Functional Requirements	38](#non-functional-requirements)

[3.5. User Interface Prototype	40](#user-interface-prototype)

[3.6. System Design	40](#system-design)  
[3.6.1.	Architectural Design	41](#architectural-design)  
[3.6.2.	Class Diagram	41](#class-diagram)  
[3.6.3.	Sequence Diagram	42](#sequence-diagram)  
[3.6.4.	Component Interaction Design	42](#component-interaction-design)  
[3.6.5.	Data Flow Architecture	43](#data-flow-architecture)  
[3.6.6.	Use Case Diagram	44](#use-case-diagram)

[3.7. Summary	45](#summary-2)

[***4\. Implementation	48***](#implementation)

[4.1. Introduction	48](#introduction-4)

[4.2. Programming Languages and Tools	48](#programming-languages-and-tools)

[4.3. User Interface Implementation	49](#user-interface-implementation)

[4.4. Database Implementation	51](#database-implementation)

[4.5. Packages and Classes Description	52](#packages-and-classes-description)

[4.6. Procedures Description	56](#procedures-description)

[4.7. Summary	62](#summary-3)

[***5\. Testing and Results	65***](#testing-and-results)

[5.1. Introduction	65](#introduction-5)

[5.2. System Testing	65](#system-testing)  
[5.2.1.	Unit Testing	65](#unit-testing)  
[5.2.2.	Integration Testing	66](#integration-testing)  
[5.2.3.	Performance Testing	66](#performance-testing)  
[5.2.4.	Test Cases	67](#test-cases)

[5.3. Discussion	68](#discussion)

[5.4. Summary	69](#summary-4)

[***6\. Conclusion and Future Work	72***](#conclusion-and-future-work)

[6.1. Conclusion	72](#conclusion)

[6.2. Future Work	73](#future-work)

[***References	75***](#references)

# **List of Figures** {#list-of-figures}

[Figure 1: Proposed User Interface Wireframes	40](#figure-1:-proposed-user-interface-wireframes)  
[Figure 2: High-level system architecture showing mobile application, backend services and ML model integration	41](#figure-2:-high-level-system-architecture-showing-mobile-application,-backend-services-and-ml-model-integration)  
[Figure 3: Class diagram of the main backend components, showing inheritance, key methods, and relationships between core classes in the Qurra system	42](#figure-3:-class-diagram-of-the-main-backend-components,-showing-inheritance,-key-methods,-and-relationships-between-core-classes-in-the-qurra-system)  
[Figure 4: Sequence diagram illustrating the reciter identification workflow, showing the interactions between the user, mobile app, backend API, audio processing, model loading, and machine learning models in the Qurra system	42](#figure-4:-sequence-diagram-illustrating-the-reciter-identification-workflow,-showing-the-interactions-between-the-user,-mobile-app,-backend-api,-audio-processing,-model-loading,-and-machine-learning-models-in-the-qurra-system)  
[Figure 5: Detailed component interactions and data flow between system modules	43](#figure-5:-detailed-component-interactions-and-data-flow-between-system-modules)  
[Figure 6: Data flow diagram illustrating audio processing pipeline from input to recognition results	44](#figure-6:-data-flow-diagram-illustrating-audio-processing-pipeline-from-input-to-recognition-results)  
[Figure 7: Use case diagram showing functional interactions between system actors and the Qurra application components	45](#figure-7:-use-case-diagram-showing-functional-interactions-between-system-actors-and-the-qurra-application-components)

# **List of Tables** {#list-of-tables}

[Table ‎1.1: Development phases and key milestones for the Qurra project implementation	25](#table-‎1.1:-development-phases-and-key-milestones-for-the-qurra-project-implementation)  
[Table ‎2.1: The Diversity of Qur'ān Recitation Styles	31](#table-‎2.1:-the-diversity-of-qur'ān-recitation-styles)  
[Table ‎4.1: Key packages and libraries used in the Qurrā system implementation	52](#table-‎4.1:-key-packages-and-libraries-used-in-the-qurrā-system-implementation)  
[Table ‎5.1: Performance Testing Results	65](#table-‎5.1:-performance-testing-results)  
[Table ‎5.2: Test Case Results and Validation	65](#table-‎5.2:-test-case-results-and-validation)  
**Chapter One: Introduction**

1. # **Introduction** {#introduction}

   1. ## **Introduction** {#introduction-1}

In recent years, the integration of AI and Islamic studies has led to innovative applications that enhance the learning and appreciation of Quranic recitation. For Muslims worldwide, the art of Tajweed is not just about correct pronunciation but also about the melodious and spiritually moving way in which verses are delivered. Each renowned reciter brings their unique style and interpretation while adhering to the established rules of Tajweed. This makes an automatic identification of Quran reciters and verses a significant challenge that combines sophisticated audio processing with advanced machine learning techniques \[1\].

This project presents \*Qurra\*, a comprehensive mobile application that can identify both Quran reciters and specific verses (Ayahs) from audio input, similar to how Shazam identifies songs. While there exist solutions like Tarteel which focuses on recitation error detection \[2\] and Nadyy, an attempted reciter identification mobile application which is now discontinued \[3\], there remains a need for a modern, accurate, and user-friendly solution that addresses both reciter and verse identification. Current systems often struggle with variations in recitation speed, background noise, and the subtle differences between similar recitation styles \[4\], \[5\].

The system employs advanced audio processing techniques and dual machine learning models to analyze recitations. Our approach combines traditional MFCC feature extraction with both Random Forest classification and Bidirectional Long Short-Term Memory (BLSTM) networks to achieve high accuracy across different recording conditions. Additionally, the system integrates Tarteel AI's fine-tuned version of Open AI's Whisper model for Arabic speech recognition, enabling accurate Ayah identification through advanced transcription and fuzzy matching techniques \[6\], \[7\]. While the project initially focuses on reciter identification, it has potential extensions to include Surah/Ayah detection and recitation style (Qirāʾāt) recognition, ultimately contributing to the preservation and accessibility of this important Islamic art form \[8\], \[9\].

2. ## **Problem Definition** {#problem-definition}

The ideal scenario is one where Muslims worldwide can easily identify both Quran reciters and specific verses they hear, whether from recordings, audio clips, or live recitations, helping them follow and learn from their preferred reciters while understanding the content being recited. However, the reality presents several significant challenges that must be addressed:

1. **Limitations of Existing Solutions:** Current solutions demonstrate considerable limitations in their approaches. Applications like Nadyy, which provided reciter identification, are now outdated or non-functional, leaving a gap in the market for modern recognition tools. Current solutions like Tarteel have shifted focus to recitation correction rather than identification. Furthermore, existing research shows that traditional machine learning methods achieved accuracy rates of only 86.5% under ideal conditions with clean audio samples, dropping significantly to below 60% when dealing with noisy or low-quality recordings \[10\]. Moreover, no comprehensive solution exists that combines both reciter identification and verse recognition in a single, user-friendly application.  
2. **Technical Challenges:** The technical challenges in Quran recognition are particularly complex due to the unique nature of Quranic recitations. Reciters employ three different speeds of recitation: Hadr (fast), Tahqeeq (slow), and Tadweer (medium), making it difficult for recognition systems to maintain accuracy across varying temporal patterns. Additionally, Tajweed rules modify letter pronunciations significantly, such as the Iqlab (إقلاب) rule changing pronunciation of "ن" to "م". These variations, combined with the inherent complexity of Arabic phonetics and the need for extensive preprocessing and feature extraction, present substantial technical hurdles \[4\], \[6\].  
3. **User Experience Gaps:** Current solutions lack the seamless cross-platform integration that users expect from modern mobile applications. Complex backend processing requirements often result in slow response times and poor performance on mobile devices. Moreover, these applications frequently fail to provide comprehensive accessibility features and lack the dual functionality of both reciter and verse identification, limiting their usefulness for comprehensive Quranic study.

The consequences of not addressing these challenges extend beyond mere technological inconvenience. Without effective recognition tools, students of the Quran face increased difficulty in identifying and following their preferred reciters while also struggling to identify specific verses they hear, potentially impacting their learning journey. The absence of reliable technology in this space creates barriers to Quranic education and appreciation, particularly for new learners who could benefit from understanding different recitation styles. Furthermore, it represents a missed opportunity to leverage modern technology in preserving and promoting this important aspect of Islamic studies \[8\], \[10\].

3. ## **Project Scope** {#project-scope}

The **Qurrā** project encompasses the following implemented components:

1. **Core Features:**  
   * Audio processing pipeline for both recorded and uploaded files  
   * Real-time reciter identification with confidence scoring  
   * Ayah (verse) identification using advanced Arabic speech recognition  
   * Cross-platform mobile application support  
   * Dual recognition capabilities in a single interface  
2. **Technical Implementation:**  
   * React Native mobile application development  
   * Python Flask backend implementation  
   * Dual machine learning model development (Random Forest and BLSTM)  
   * Tarteel Whisper model integration for Arabic speech recognition  
   * Comprehensive API development for model integration  
   * Advanced fuzzy matching algorithms for verse identification  
3. **Data Management:**  
   * Dataset creation from everyayah.com/data for reciter training  
   * Comprehensive Quran verse database integration  
   * Audio preprocessing pipeline with noise reduction  
   * Advanced feature extraction systems  
   * Model training, validation, and performance optimization  
4. **Delivered Capabilities:**  
   * Reciter identification with \>90% accuracy  
   * Ayah identification with fuzzy matching support  
   * Confidence scoring for both recognition types  
   * Cross-platform mobile deployment  
5. **Non-Goals:**  
   * Offline recognition capabilities (reserved for future versions)  
   * Multi-language support (Arabic and English)  
   * Qirāʾāt detection (reserved for future enhancements)  
   * Multilingual translation services (reserved for future enhancements)

   4. ## **Local Impact** {#local-impact}

The Qurra system aligns with Saudi Arabia's Vision 2030 by applying advanced technologies to preserve and enhance Quranic education and cultural heritage. The project contributes to digital transformation in religious education while supporting the Kingdom's goals of innovation and cultural preservation.

In this context, the project has the potential to make a significant impact in Saudi Arabia and the broader Middle East region in the following ways:

1. **Enhancing Quranic Education:** The system serves as a transformative tool for students and educators in Quranic schools. By enabling users to identify both reciters and specific verses, it complements traditional teaching methods and fosters more engaging, interactive learning experiences. This supports Vision 2030's goals of educational innovation and improved teaching practices.  
2. **Preserving Cultural Heritage:** The system contributes to preserving Quranic recitation traditions by documenting and identifying different recitation styles and renowned reciters. This digital preservation aligns with Vision 2030's emphasis on sustaining Saudi Arabia's rich cultural heritage while ensuring that traditional knowledge is accessible to future generations \[11\].  
3. **Fostering Technological Innovation:** By applying machine learning, natural language processing, and mobile technologies to Islamic studies, the project demonstrates innovative technological approaches that align with Vision 2030's vision of a knowledge-based economy and technological advancement \[12\].  
4. **Supporting Arabic Language Processing:** The system's Arabic speech recognition capabilities contribute significantly to Arabic language technology development. The techniques developed can be adapted for other Arabic processing applications, supporting broader technological advancement in the region \[13\].

   5. ## **Global Impact** {#global-impact}

On a global scale, the Qurra system has significant potential impact:

1. **Enhancing Global Accessibility:** By providing comprehensive Quran recognition capabilities, the system enhances accessibility for Muslims worldwide, particularly those in non-Arabic-speaking countries where access to traditional Quranic education may be limited. The dual functionality of reciter and verse identification provides unprecedented learning support \[14\].  
2. **Facilitating Comprehensive Quranic Education:** The system can be integrated into global Quranic education initiatives, providing tools for learners to engage with authentic recitations while understanding the content being recited. This deepens both appreciation of recitation styles and comprehension of Quranic content \[15\].  
3. **Contributing to Digital Islamic Humanities:** The project represents a significant contribution to digital Islamic humanities by applying advanced computing techniques to both recitation analysis and verse recognition, opening new avenues for Quranic research and study \[16\].  
4. **Setting New Standards:** As a comprehensive solution combining reciter identification and verse recognition, the project sets new benchmarks for Quranic technology applications, inspiring future developments in Islamic educational technology \[17\].

   6. ## **Aims and Objectives** {#aims-and-objectives}

**Aim:** To develop an intelligent mobile application that accurately identifies both Quran reciters and verses from audio input, advancing the accessibility and preservation of authentic Quran recitations and content through modern technology.

**Objectives:**

1. **Collect and prepare comprehensive datasets that:**  
   * Include diverse recordings from mp3quran.net for reciter training  
   * Integrate complete Quran verse database for Ayah identification  
   * Support both reciter and verse recognition model training  
   * Ensure data quality and preprocessing standards  
2. **Implement dual ML systems that:**  
   * Achieve \>90% accuracy in reciter identification using Random Forest and BLSTM models  
   * Provide reliable Ayah identification through Whisper model integration  
   * Process inputs in under 5 seconds for real-time user experience  
   * Maintain robustness with a minimum of 85% accuracy for noisy audio  
   * Deliver confidence scores for both recognition types  
3. **Create a robust backend system that:**  
   * Processes audio inputs efficiently through Flask server architecture  
   * Implements secure API endpoints for both recognition services  
   * Manage comprehensive reciter and verse databases effectively  
   * Integrates multiple ML models seamlessly  
4. **Develop a cross-platform mobile application that:**  
   * Provides intuitive interface for both reciter and Ayah identification  
   * Displays results clearly with confidence scores and detailed information  
   * Ensures responsive performance across iOS and Android devices  
   * Supports multiple languages and accessibility features

   7. ## **Alternative Solutions** {#alternative-solutions}

The field of Quran audio recognition has explored various approaches, each with distinct advantages and limitations:

1. **Deep Learning Approaches:** Advanced models utilizing CNN-Bidirectional GRU architectures have achieved up to 98% accuracy on clean datasets. These models excel at capturing complex patterns but require substantial computational resources and large datasets, making mobile deployment challenging \[18\].  
2. **Traditional ML Approaches:** Support Vector Machines (SVM) and Hidden Markov Models have been widely used due to their effectiveness with smaller datasets and lower computational requirements. However, performance declines significantly with noisy or diverse audio conditions \[18\].  
3. **Audio Fingerprinting:** Systems like the discontinued Nadyy application used audio fingerprinting for rapid identification. While offering fast response times, these approaches are heavily reliant on exact database matches and struggle with recitation variations.  
4. **Hybrid Approaches:** Recent research has explored combining multiple techniques, such as MFCC features with various classifiers, achieving balanced accuracy and efficiency \[5\].

**Our Implemented Solution:**

Based on evaluation of these alternatives and project requirements, we implemented a comprehensive hybrid approach that integrates:

* **Dual ML Models:** Random Forest for efficient processing and BLSTM for enhanced accuracy  
* **Advanced Feature Extraction:** MFCC-based audio processing with noise reduction  
* **Arabic Speech Recognition:** Tarteel Whisper model for verse identification  
* **Intelligent Matching:** Fuzzy matching algorithms for robust verse recognition  
* **Mobile-Optimized Architecture:** Flask backend designed for mobile application support

This approach balances accuracy, computational efficiency, and comprehensive functionality while supporting both reciter and verse identification in a single system.

8. ## **Method / Approach** {#method-/-approach}

The development of Qurra follows a systematic approach structured around four primary components: **Frontend Development**, **Backend Development**, **ML Model Development**, and **Speech Recognition Integration**.

**ML Model Development:**

The ML implementation employs a dual-model approach for optimal performance:

1. **Random Forest Implementation:**  
   * MFCC feature extraction for audio signal processing  
   * Random Forest classification for efficient reciter identification  
   * Calibrated probability outputs for confidence scoring  
   * Cross-validation for model optimization  
2. **BLSTM Implementation:**  
   * Bidirectional LSTM architecture for temporal pattern recognition  
   * Advanced audio preprocessing with noise reduction  
   * Attention mechanisms for improved accuracy  
   * Standard GPU optimization for faster training and inference when available  
3. **Performance Targets:**  
   * \> 90% accuracy on test datasets  
   * Response times under 5 seconds  
   * Robust performance across various audio conditions

**Speech Recognition Integration:**

For Ayah identification, the system integrates:

* **Tarteel Whisper Model:** Arabic speech recognition optimized for Quranic recitation  
* **Text Normalization:** Advanced Arabic text processing for accurate matching  
* **Fuzzy Matching:** Intelligent algorithms to handle partial or imperfect transcriptions  
* **Verse Database:** Comprehensive Quran verse storage with efficient retrieval

**Backend Development:**

The backend utilizes **Python’s Flask** framework providing:

* RESTful API endpoints for both reciter and Ayah identification (/getReciter, /getAyah)  
* Additional endpoints for reciter listing (/getAllReciters) and health monitoring (/health)  
* Efficient model loading and inference management  
* Advanced audio processing pipelines  
* Secure file handling and user data management  
* Configuration-based system management

**Frontend Development:**

React Native implementation ensures:

* Cross-platform compatibility (iOS and Android)  
* Intuitive user interface for dual recognition capabilities  
* Real-time audio recording and file upload support  
* Responsive design with confidence score visualization  
* Multi-language support (Arabic and English)

**System Integration:**

The complete system integrates through:

* Secure API communication between mobile app and Flask server  
* Efficient model orchestration for dual recognition services  
* Advanced error handling and user feedback systems  
* Performance monitoring and optimization

  9. ## **Project Timeline** {#project-timeline}

The project was implemented over multiple phases as detailed in Table 1.1, which outlines the development phases, key activities, deliverables, and associated risks for each stage of the Qurra project implementation

*Table ‎1.1: Development phases and key milestones for the Qurra project implementation*

| Phase | Duration | Activities | Deliverables | Dependencies/Risks |
| :---- | :---- | :---- | :---- | :---- |
| Dataset Preparation | 1 week | Data collection from mp3quran.netQuran verse database integrationData cleaning and preprocessingFeature extraction pipeline development | Preprocessed reciter datasetComplete Quran verse databaseDocumented preprocessing pipeline | **Risk:** Limited diversity in dataset**Mitigation:** Augment data with synthetic samples |
| ML Model Development | 5 weeks | Implementation of Random Forest classifierDevelopment of BLSTM architectureTraining and validation of both modelsPerformance benchmarking and optimization | Trained Random Forest modelOptimized BLSTM modelPerformance metrics documentationModel comparison analysis | **Risk:** Overfitting or poor generalization**Mitigation:** Cross-validation and regularization techniques |
| Speech Recognition Integration | 3 weeks | Tarteel Whisper model integrationArabic text processing developmentFuzzy matching algorithm implementationVerse database optimization | Functional Ayah identification systemOptimized matching algorithmsComprehensive verse database | **Risk:** Arabic transcription accuracy**Mitigation:** Advanced text normalization |
| Backend Development | 4 weeks | Flask server implementationAPI endpoint developmentModel integration and optimizationSecurity and authentication setup | Functional Flask backendComplete API documentationIntegrated dual ML systemsSecurity implementation | **Risk:** Model loading performance**Mitigation:** Efficient caching and optimization |
| Frontend Development | 4 weeks | React Native application developmentUI/UX design and implementationAPI integration and testingCross-platform optimization | User-friendly mobile applicationTested frontend-backend integrationCross-platform compatibility | **Risk:** Performance on low-end devices**Mitigation:** Optimization and progressive loading |
| Testing & Optimization | 2 weeks | Comprehensive system testingPerformance optimizationUser acceptance testingBug fixing and refinement | Optimized and tested systemUser feedback documentationPerformance benchmarks | **Risk:** Unresolved critical issues**Mitigation:** Systematic testing protocols |

As shown in Table 1.1, the total development timeline spans 19 weeks, with ML model development requiring the longest duration due to the complexity of training and optimizing dual recognition systems. The overlapping phases allow for parallel development where dependencies permit, optimizing overall project completion time.

10. ## **Report Structure** {#report-structure}

This report is organized into the following chapters:

1. **Chapter 1: Introduction**: Presents project overview, problem definition, objectives, and methodology  
2. **Chapter 2: Literature Review**: Examines existing solutions, reviews relevant research in audio processing and ML, analyzes gaps in current implementations  
3. **Chapter 3: System Analysis and Design**: Details system requirements and architecture, presents design specifications, describes implementation approach  
4. **Chapter 4: Implementation**: Documents programming languages and tools, describes UI and system implementation, explains packages and procedures  
5. **Chapter 5: Testing and Results**: Presents system testing methodologies, discusses results and performance metrics, analyzes findings  
6. **Chapter 6: Conclusion and Future Work**: Summarizes achievements, discusses lessons learned, outlines future enhancements

   11. ## **Summary** {#summary}

This chapter has introduced the Qurra project, a comprehensive mobile application for both Quran reciter and verse identification. We have outlined the problem definition, highlighting the need for a modern, dual-functional system that addresses limitations in existing solutions. The project successfully delivers both reciter identification through dual ML models (Random Forest and BLSTM) and Ayah identification through advanced Arabic speech recognition.

The implemented solution combines traditional machine learning with deep learning approaches and state-of-the-art speech recognition, delivered through a React Native mobile application with a robust Flask backend. The system achieves the target performance metrics while providing users with unprecedented capabilities in Quranic audio recognition.

The next chapter will present a comprehensive literature review, examining existing research and technologies in audio processing, machine learning, and Quranic recitation recognition.

**Chapter Two: Literature Review**

2. # **Literature Review** {#literature-review}

   1. ## **Introduction** {#introduction-2}

This chapter provides a comprehensive review of technologies, methodologies, and research developments in Quran reciter recognition and Arabic speech recognition systems. The review encompasses both traditional approaches and modern deep learning techniques, examining their effectiveness and limitations. We begin with fundamental background information, followed by an analysis of related work in audio processing, machine learning models, and Arabic speech recognition systems that inform both reciter identification and verse recognition capabilities.

The evolution of Quran audio recognition technologies demonstrates the field's progression from simple pattern matching to sophisticated deep learning approaches, now extending to comprehensive systems that combine reciter identification with verse recognition through advanced Arabic speech processing.

2. ## **Background** {#background}

   1. ### **The Significance of Qur’ānic Audio Processing** {#the-significance-of-qur’ānic-audio-processing}

Quran recitation holds fundamental importance in Islamic practice, representing both spiritual devotion and artistic expression. Muslims believe that the Quran should be recited beautifully and melodiously, as it was revealed to Prophet Muhammad (ﷺ). The art of Quran recitation encompasses not only correct pronunciation of Arabic words but also proper application of Tajweed rules, which govern the manner of recitation \[19\].

The Prophet Muhammad (ﷺ) emphasized the importance of beautiful recitation, saying: "Make your voices beautiful when you recite Quran" \[20\]. This tradition has led to the development of diverse recitation styles, with renowned reciters leaving lasting legacies through their unique and captivating recitations that serve as models for learning and spiritual reflection.

2. ### **The Diversity of Qur’ān Recitation Styles** {#the-diversity-of-qur’ān-recitation-styles}

Quran recitation is distinguished by its recognized styles, known as Qirāʾāt, representing variations in pronunciation, elongation, and articulation rooted in authentic transmission chains from Prophet Muhammad (ﷺ). This oral tradition has been meticulously preserved over centuries to ensure accessibility while maintaining authenticity \[21\].

The ten recognized Qirāʾāt include those of Nafi', Ibn Kathir, Abu Amr, Ibn Amir, Asim, Hamzah, Al-Kisai, Abu Jaafar, Yaqub, and Khalaf. Each follows unique Tajweed rules governing letter pronunciation, vowel elongation, and rhythmic variations. These differences enrich the Quran's oral tradition without altering meaning, reflecting Arabic language flexibility while preserving universal message \[22\].

The Hafs narration from Asim is most widely practiced worldwide, serving as the standard recitation in most printed Qurans and Islamic educational institutions, making it a primary focus for recognition systems. Table 2.1 shows the global distribution of major Quranic recitation styles, clearly showing the dominance of the Hafs transmission which accounts for 95% of global usage.

*Table ‎2.1: The Diversity of Qur'ān Recitation Styles*

| Qirāʾāt | Transmission | Geographic Distribution | Global Usage (%) |
| :---: | :---: | :---: | :---: |
| 'Asim | Hafs | The Muslim world in general | 95% |
| Nafi' | Warsh | Algeria, Morocco, parts of Tunisia, West Africa and Sudan | 3% |
| Nafi' | Qalun | Libya, Tunisia and parts of Qatar | 0.7% |
| Abu 'Amr al-Ala | Al-Duri | Parts of Sudan and West Africa | 0.3% |
| Ibn Amir | Hisham/Ibn Dhakwan | Parts of Yemen | 1% |

3. ### **Audio Processing in Islamic Applications** {#audio-processing-in-islamic-applications}

Application of audio processing technology to Islamic content, particularly Quran recitations, presents unique challenges. Quranic recitation follows Tajweed rules and can be performed in multiple styles, making recognition more complex than standard speaker identification \[23\].

Technical challenges include significant acoustic feature variations depending on reciter voice, style, and recording environment \[4\]. Additionally, the linguistic and phonetic characteristics of Quranic Arabic differ from modern standard Arabic, necessitating specialized audio processing and machine learning techniques \[5\].

4. ### **Machine Learning and Deep Learning Approaches** {#machine-learning-and-deep-learning-approaches}

The field has evolved from traditional machine learning to sophisticated deep learning architectures:

1. **Traditional Methods:**  
   * Hidden Markov Models (HMM) for sequence modeling  
   * Support Vector Machines (SVM) for classification  
   * Random Forest for ensemble learning with good interpretability  
   * k-Nearest Neighbors for similarity-based classification  
2. **Deep Learning Methods:**  
   * Convolutional Neural Networks (CNNs) for spectral pattern recognition  
   * Recurrent Neural Networks (RNNs) for temporal sequence modeling  
   * Bidirectional LSTM for enhanced temporal context understanding  
   * Hybrid architecture combining multiple deep learning approaches  
3. **Arabic Speech Recognition:**  
   * Transformer-based models for language understanding  
   * Whisper models specifically trained for Arabic speech  
   * End-to-end speech recognition systems  
   * Domain-specific models for Quranic Arabic processing

   3. ## **Related Work** {#related-work}

      1. ### **Traditional Approaches to Reciter Recognition** {#traditional-approaches-to-reciter-recognition}

Early Quran reciter identification research relied on conventional machine learning with handcrafted features. Hussaini and Aldhaheri \[10\] introduced foundational work using MFCC features with k-Nearest Neighbors classification, achieving 86% accuracy on clean datasets. However, this approach struggled with scalability and noise sensitivity, as k-NN is vulnerable to outliers and requires careful distance metric selection.

Nahar et al. \[5\] developed an efficient system using Support Vector Machines with MFCC features, implementing a three-stage model: acoustic feature extraction, SVM training on labeled categories, and classification. Their system achieved 96% accuracy, outperforming Decision Trees and Naïve Bayes classifiers. Despite high accuracy, reliance on pre-defined features limited generalizability to larger, more diverse datasets.

2. ### **Deep Learning Approaches for Reciter Recognition** {#deep-learning-approaches-for-reciter-recognition}

Modern deep learning methods have significantly advanced reciter recognition through automatic feature learning. Nahar et al. \[5\] developed CNN-based models leveraging MFCC and GTCC features to capture spectral and spatial information, achieving 96% accuracy. Convolutional layers proved effective for analyzing audio spectrograms, though computational costs and large dataset requirements posed deployment challenges.

Qayyum et al. \[8\] introduced hybrid architectures combining pre-trained VGG models for feature extraction with BLSTM layers for sequence modeling. This approach achieved state-of-the-art 99% accuracy, demonstrating the effectiveness of combining convolutional and recurrent architectures for capturing both spatial and temporal dependencies in Quranic recitations. However, substantial computational requirements limit mobile application deployment.

3. ### **Arabic Speech Recognition Systems** {#arabic-speech-recognition-systems}

Arabic speech recognition, particularly for Quranic content, presents unique challenges due to dialectal variations, complex morphology, and specialized vocabulary. Recent advances in transformer-based models, particularly Whisper architecture, have shown significant improvements in Arabic speech recognition accuracy \[13\].

Tarteel AI has developed specialized Whisper models optimized for Quranic Arabic, addressing the specific phonetic and linguistic characteristics of religious recitation. These models demonstrate superior performance in transcribing Quranic verses compared to general-purpose Arabic speech recognition systems, making them ideal for verse identification applications.

4. ### **Hybrid and Multi-Modal Approaches** {#hybrid-and-multi-modal-approaches}

Recent research explores combining multiple recognition modalities and techniques. Transfer learning approaches utilize pre-trained models like ResNet for feature extraction, followed by fine-tuning on Quranic recitation data, achieving 98% accuracy while reducing training time \[4\].

Multi-modal learning, integrating audio with textual metadata, has shown promise in related applications. While limited implementation exists specifically for Quran recognition, successful applications in general speech and emotion recognition provide promising directions for future development \[5\].

5. ### **Challenges and Limitations in Current Systems** {#challenges-and-limitations-in-current-systems}

Despite significant advancements, several challenges persist:

1. **Dataset Limitations:**  
   * Existing datasets often lack diversity in recitation styles and recording conditions  
   * Limited representation of different Qirāʾāt and regional variations  
   * Insufficient noisy or real-world recording examples for robust training  
2. **Computational Constraints:**  
   * High-performance deep learning models require substantial computational resources  
   * GPU requirements limit deployment of resource-constrained mobile devices  
   * Real-time processing demands conflict with complex model requirements  
3. **Integration Challenges:**  
   * Most systems focus on single-task recognition (either reciter or content identification)  
   * Limited integration between reciter recognition and speech recognition systems  
   * Lack of comprehensive solutions combining multiple recognition capabilities  
4. **Arabic Speech Processing:**  
   * Dialectal variations in pronunciation affect recognition accuracy  
   * Complex Arabic morphology requires sophisticated text processing  
   * Limited availability of large-scale Arabic speech datasets for training

   4. ## **Summary** {#summary-1}

This literature review reveals significant progress in Quran reciter recognition, from traditional machine learning approaches achieving 86-96% accuracy to advanced deep learning systems reaching 99% accuracy under controlled conditions. Key findings include:

* **Feature Extraction Evolution:** MFCC remains fundamental, but integration with deep learning features shows superior performance  
* **Model Architecture Advancement:** Hybrid approaches combining CNN and BLSTM architecture demonstrate optimal balance of spatial and temporal pattern recognition  
* **Arabic Speech Recognition Progress:** Specialized models like Tarteel Whisper show significant improvements in Quranic Arabic transcription accuracy  
* **Integration Opportunities:** Limited work exists on comprehensive systems combining reciter identification with verse recognition capabilities

**Research Gaps Identified:**

* Lack of comprehensive solutions integrating both reciter and verse identification  
* Limited mobile-optimized implementations balancing accuracy with computational efficiency  
* Insufficient focus on real-world deployment challenges and user experience considerations  
* Need for standardized evaluation metrics and benchmark datasets

These insights inform the design of comprehensive systems that address both reciter identification and verse recognition while maintaining practical deployment feasibility. The next chapter presents our system analysis and design approach, building upon these literature findings to create an integrated solution addressing identified gaps.

**Chapter Three: System Analysis and Design**

3. # **System Analysis and Design** {#system-analysis-and-design}

   1. ## **Introduction** {#introduction-3}

This chapter presents the detailed analysis and design of the Qurra system, a comprehensive Quran reciter and verse identification application. The design integrates dual recognition capabilities through advanced machine learning models and Arabic speech recognition technology. We describe both functional and non-functional requirements, along with architectural design decisions influenced by literature review findings and mobile deployment requirements.

2. ## **System Perspective** {#system-perspective}

The Qurra system operates as a mobile application with backend support, integrating several key components for comprehensive Quran audio recognition:

1. **Mobile Application Layer**  
   * Cross-platform React Native frontend  
   * Dual-purpose audio capture (recording and file upload)  
   * Multi-language user interface components  
   * Real-time result visualization with confidence scoring  
2. **Backend Services Layer**  
   * Python Flask RESTful API endpoints  
   * Dual ML model deployment and orchestration  
   * Advanced audio preprocessing pipelines  
   * Arabic speech recognition integration  
3. **Machine Learning Layer**  
   * Random Forest classifier for efficient reciter identification  
   * BLSTM model for enhanced accuracy recognition  
   * Feature extraction and preprocessing pipelines  
   * Model performance monitoring and optimization  
4. **Speech Recognition Layer**  
   * Tarteel Whisper model integration for Arabic transcription  
   * Text normalization and preprocessing  
   * Fuzzy matching algorithms for verse identification  
   * Comprehensive Quran verse database management

The system maintains independence in core functionality while integrating external speech recognition services for enhanced capabilities.

3. ## **Requirements Elicitation Techniques** {#requirements-elicitation-techniques}

Requirements were gathered through comprehensive analysis:

1. **Literature Analysis**  
   * Examined existing Quran recognition solutions and their limitations  
   * Reviewed technical research in audio processing, machine learning, and Arabic speech recognition  
   * Analyzed performance benchmarks and user experience requirements  
2. **Technical Feasibility Study**  
   * Evaluated machine learning model performance and computational requirements  
   * Assessed mobile deployment constraints and optimization needs  
   * Determined optimal architecture for dual recognition capabilities  
3. **User Experience Research**  
   * Analyzed user needs for both reciter and verse identification  
   * Studied accessibility requirements for diverse user populations  
   * Assessed cross-platform mobile application requirements

   4. ## **System Requirements** {#system-requirements}

      1. ### **Functional Requirements** {#functional-requirements}

The system must fulfill comprehensive recognition functionalities:

1. **Audio Input Management:**  
   * System shall allow users to record live recitations using device microphones  
   * System shall support audio file uploads in multiple formats (MP3, WAV, M4A)  
   * System shall perform real-time audio quality validation and preprocessing  
   * System shall handle audio duration constraints (5-15 seconds for both reciter ID and Ayah ID)  
   * Reciter Recognition Processing:  
   * System shall identify reciter using dual ML models (Random Forest and BLSTM)  
   * System shall achieve \>90% accuracy in reciter identification  
   * System shall provide confidence scores indicating model certainty  
   * System shall process reciter identification requests efficiently  
   * System shall support model comparison and selection for optimal performance  
2. **Ayah Recognition Processing:**  
   * System shall transcribe Arabic audio using Tarteel Whisper model  
   * System shall identify specific Quran verses through fuzzy matching  
   * System shall handle partial or imperfect recitations with confidence scoring  
   * System shall provide verse text, Surah information, and location details  
3. **Results Management:**  
   * System shall display comprehensive results for both recognition types  
   * System shall provide reciter profiles with biographical information and sample audio  
   * System shall show verse details including Arabic text, English transliteration, and Surah context  
   * System shall maintain result history for user reference  
   * System shall support result sharing through various communication channels  
4. **API Endpoints**  
   * System shall provide /getReciter endpoint for reciter identification  
   * System shall provide /getAyah endpoint for Ayah identification  
   * System shall provide /getAllReciters endpoint for reciter listing  
   * System shall provide /health endpoint for system health monitoring

     2. ### **Non-Functional Requirements** {#non-functional-requirements}

1. **Performance Requirements:**  
   * System must process recognition requests within 5 seconds for optimal user experience  
   * Backend shall maintain 99.9% uptime ensuring reliable service availability  
   * System should support up to 1000 concurrent users during peak usage  
   * Mobile application memory usage should not exceed 100MB on target devices  
   * Audio processing should maintain quality while optimizing mobile bandwidth  
2. **Accuracy Requirements:**  
   * Reciter identification must achieve a minimum 90% accuracy on test datasets  
   * System must maintain a minimum of 85% accuracy with noisy or low-quality audio  
   * Ayah identification should achieve a minimum of 70% confidence matching for fuzzy recognition  
   * System must provide reliable confidence scores for both recognition types  
3. **Security Requirements:**  
   * All data transmissions must use HTTPS encryption for secure communication  
   * User audio data must be processed securely with optional local deletion  
   * API endpoints must implement proper authentication and rate limiting  
   * System must comply with data privacy regulations and user consent requirements  
4. **Usability Requirements:**  
   * User interface must be intuitive with minimal learning curve  
   * System shall support Arabic and English language interfaces  
   * Application must provide clear error messages and user guidance  
   * Interface should be responsive across various mobile device screen sizes  
   * System must provide accessibility features for users with different abilities  
5. **Scalability Requirements:**  
   * Backend architecture must support horizontal scaling for increased user load  
   * System must allow model updates without service interruption  
   * Efficient caching mechanisms must reduce processing load and improve response times  
   * Architecture should support future feature additions without major

   5. ## **User Interface Prototype** {#user-interface-prototype}

Figure 1 presents the proposed user interface wireframes for the Qurra mobile application, showcasing the key screens and user interaction flows.

| ![][image3] | ![][image4] |
| :---: | :---: |
| ![Screens screenshot of a video gameAI-generated content may be incorrect.][image5] | ![][image6] |

*Figure 1: Proposed User Interface Wireframes*

6. ## **System Design** {#system-design}

   1. ### **Architectural Design** {#architectural-design}

The system architecture follows a layered approach that separates concerns while enabling efficient communication between components. Figure 2 illustrates the high-level system architecture, showing how the mobile application layer interfaces with backend services and ML model integration. The architecture demonstrates the clear separation between the React Native frontend, Flask backend services, and the underlying machine learning and speech recognition layers.

![][image7]

*Figure 2: High-level system architecture showing mobile application, backend services and ML model integration*

2. ### **Class Diagram** {#class-diagram}

The object-oriented design of the backend system is represented through key classes and their relationships. Figure 3 presents the class diagram of the main backend components, illustrating inheritance patterns, key methods, and relationships between core classes in the Qurra system. The diagram shows how the BaseModel class serves as the foundation for both RandomForestModel and BLSTMModel implementations, while service classes handle the business logic for reciter and Ayah identification.

![A diagram of a networkAI-generated content may be incorrect.][image8]

*Figure 3: Class diagram of the main backend components, showing inheritance, key methods, and relationships between core classes in the Qurra system*

3. ### **Sequence Diagram** {#sequence-diagram}

The interaction flow between system components during the reciter identification process is detailed in Figure 4\. This sequence diagram illustrates the complete workflow from user audio input through to result delivery, showing the interactions between the user, mobile app, backend API, audio processing components, model loading mechanisms, and machine learning models. The diagram demonstrates the asynchronous nature of the processing pipeline, and the coordination required between multiple system components.

![A diagram with text and linesAI-generated content may be incorrect.][image9]

*Figure 4: Sequence diagram illustrating the reciter identification workflow, showing the interactions between the user, mobile app, backend API, audio processing, model loading, and machine learning models in the Qurra system*

4. ### **Component Interaction Design** {#component-interaction-design}

The detailed interactions between system modules are further illustrated in Figure 5, which shows the component interaction design and data flow between system modules. This diagram provides a comprehensive view of how audio data flows through the preprocessing pipeline, feature extraction, model inference, and result formatting stages.

![A diagram of a diagramAI-generated content may be incorrect.][image10]

*Figure 5: Detailed component interactions and data flow between system modules*

5. ### **Data Flow Architecture** {#data-flow-architecture}

Figure 6 presents the data flow architecture, illustrating the complete audio processing pipeline from initial input to final recognition results. The diagram shows how raw audio data is transformed through various stages including preprocessing, feature extraction, model inference, and result compilation for both reciter and Ayah identification workflows.

![A diagram of a companyAI-generated content may be incorrect.][image11]

*Figure 6: Data flow diagram illustrating audio processing pipeline from input to recognition results*

6. ### **Use Case Diagram** {#use-case-diagram}

The use case design defines the functional interactions between system actors and the Qurra application. Figure 7 presents the comprehensive use case model that captures both user-facing functionality and administrative operations required for system maintenance and optimization.

The diagram identifies two primary actors: Users who interact with the core recognition features, and Administrators who manage system resources and performance. User interactions center around the dual recognition capabilities, including audio input methods (recording and uploading), result viewing, and reciter database exploration. Administrative functions encompass database management, system monitoring, and machine learning model updates to ensure continued system accuracy and performance.

The \<\<include\>\> relationships demonstrate the natural workflow extensions where recognition activities lead to result presentation, creating an integrated user experience that supports both immediate identification needs and comprehensive exploration of Quranic recitation content.

![A screenshot of a computerAI-generated content may be incorrect.][image12]

*Figure 7: Use case diagram showing functional interactions between system actors and the Qurra application components*

7. ## **Summary** {#summary-2}

This chapter has presented the comprehensive system analysis and design for Qurra, addressing both technical requirements and user experience considerations. Key design decisions include:

* **Dual Recognition Architecture:** Integrated reciter and verse identification capabilities  
* **Hybrid Model Approach:** Combination of traditional ML and deep learning methods  
* **Mobile-First Design:** Cross-platform React Native application with efficient backend processing  
* **File-Based Data Management:** Optimized for machine learning workflows and flexible configuration  
* **Modular Architecture:** Enabling easy maintenance, updates, and feature extensions

The design addresses the complex requirements of Quranic audio processing while maintaining practical deployment constraints for mobile devices. The modular architecture supports both current functionality and future enhancements, providing a solid foundation for the implementation phase described in the next chapter.  
The architectural decisions balance accuracy, performance, and usability to create a comprehensive solution that serves the diverse needs of the global Muslim community while advancing the state of Islamic educational technology.

**Chapter Four: Implementation**

4. # **Implementation** {#implementation}

   1. ## **Introduction** {#introduction-4}

This chapter presents the detailed implementation of the Qurra system, covering the technical aspects of developing a comprehensive Quran reciter and verse identification application. The implementation encompasses three main components: a React Native mobile application for cross-platform deployment, a Python Flask backend server for API services, and sophisticated machine learning pipelines for audio recognition capabilities.

The implementation process focused on creating a robust, scalable system that balances high accuracy with practical mobile deployment requirements. The development followed modern software engineering practices, including modular architecture, comprehensive testing, and performance optimization. This chapter details the programming languages and tools used, user interface implementation strategies, data management approaches, key packages and libraries, and critical implementation procedures that ensure the system's effectiveness and reliability.

2. ## **Programming Languages and Tools** {#programming-languages-and-tools}

The Qurra system implementation utilized a diverse technology stack optimized for both mobile development and machine learning capabilities:

1. **Programming Languages**  
   * **TypeScript:** Primary language for React Native mobile application development, chosen for its cross-platform compatibility and extensive ecosystem. TypeScript provided additional type safety and improved development experience through enhanced IDE support and compile-time error detection.  
   * **Python:** Selected for backend development and machine learning implementation due to its extensive libraries for audio processing, machine learning, and scientific computing. Python's readability and rapid development capabilities made it ideal for prototyping and implementing complex ML algorithms.  
   * **HTML/CSS:** Used for web-based components and styling within the React Native application, particularly for rendering Arabic text and complex layouts.  
2. **Development Tools and Environment**  
   * **Visual Studio Code:** Primary integrated development environment (IDE) used throughout the project for both frontend and backend development. VS Code provided excellent extensions for React Native, Python, and machine learning development, including debugging capabilities and integrated terminal support.  
   * **Node.js:** JavaScript runtime environment essential for React Native development, package management, and build processes.  
   * **Python Environment:** Python 3.9+ was used with virtual environment management for dependency isolation and reproducible builds.  
   * **Git/GitHub:** Version control system used for code versioning, collaboration, branch management, and project documentation throughout the development process.  
   * **Postman:** API testing and development tool used for testing Flask endpoints, validating request/response formats, and debugging API functionality.  
3. **Hardware and Computational Resources**  
* **Local Development:** Personal development machines running Windows/macOS for application development, testing, and debugging  
* **GPU-Accelerated Training:** Google Colab Pro and partner GPU-enabled systems for intensive model training tasks, particularly for BLSTM model optimization  
* **Google Colab:** Used for GPU-accelerated model training when local resources were insufficient  
4. **Mobile Development Tools**  
* **Expo:** React Native framework used for cross-platform mobile application development, providing streamlined development workflow and easy deployment across iOS and Android platforms.

  3. ## **User Interface Implementation** {#user-interface-implementation}

The user interface implementation for Qurra emphasized intuitive design and seamless user experience across both iOS and Android platforms. The implementation strategy focused on creating a clean, accessible interface that accommodates the unique requirements of Quranic audio processing.

**Design Philosophy and Architecture**

The UI implementation followed a tab-based navigation pattern with three primary screens: Reciter Identification, Ayah Identification, and Reciter Search. This design choice was made to clearly separate the dual functionality while maintaining intuitive navigation. The interface incorporates Islamic design elements and typography to create a culturally appropriate user experience.

* **Component-Based Architecture:** The interface was built using reusable React Native components, promoting code maintainability and consistency across screens. Custom components were developed for audio recording, file upload, result display, and Arabic text rendering.  
* **State Management:** React hooks and context API were utilized for efficient state management, particularly for handling audio processing states, recognition results, and user preferences.

**Key Interface Components**

* **Audio Input Interface:** The recording interface features a prominent circular recording button with visual feedback including pulse animations during recording and timer display. The implementation includes real-time audio level monitoring and automatic stop functionality for duration limits.  
* **Dual Recognition Tabs:** Separate tabs for reciter and Ayah identification allow users to easily switch between recognition modes. Each tab is optimized for its specific use case, with appropriate duration limits and result formatting.  
* **Results Display:** Recognition results are presented with confidence scores, detailed information, and visual indicators. The interface includes circular progress indicators for confidence levels and comprehensive reciter profiles with biographical information.  
* **Arabic Text Rendering:** Special attention was given to proper Arabic text display, including right-to-left text direction, appropriate fonts for Quranic text, and Tajweed visualization for verse displays.

**Cross-Platform Considerations**

The implementation addresses platform-specific requirements while maintaining consistent functionality:

* **Audio Permissions:** Platform-specific permission handling for microphone access and file system access.  
* **File Handling:** Cross-platform file upload and audio processing capabilities with format validation.  
* **Performance Optimization:** Memory management and performance optimization for audio processing on both iOS and Android devices.

**Reasoning for Technology Choices**

React Native was chosen for mobile development due to its ability to provide native performance while maintaining a single codebase for both platforms. This choice significantly reduced development time and maintenance overhead while ensuring consistent user experience across devices. The framework's extensive ecosystem and community support provided excellent libraries for audio processing, file handling, and Arabic text rendering.

Component-based architecture using React hooks provided excellent state management capabilities essential for handling complex audio processing workflows and real-time user feedback. The modular design also enables easy maintenance and future feature additions.

4. ## **Database Implementation** {#database-implementation}

The Qurra system employs a file-based data management approach rather than traditional database systems, strategically chosen for the current development phase and specific project requirements.

**Data Storage Architecture**

* **JSON-Based Configuration:** System configuration, model parameters, and application settings are stored in structured JSON files. This approach provides flexibility for configuration management and easy deployment across different environments.  
* **Quran Verse Database:** A comprehensive JSON database contains all Quranic verses organized hierarchically by Surah and Ayah. The database includes Arabic text, transliterations, and metadata required for verse identification and display.  
* **Model Serialization:** Machine learning models are serialized using Python's joblib library for efficient loading and storage. This approach ensures fast model loading times and optimal memory usage during inference.  
* **Reciter Profiles:** Reciter information is stored in structured JSON format, including biographical details, sample audio metadata, and model training information.

**Reasoning for File-Based Approach**

The file-based approach was strategically chosen for the current development phase based on several factors:

* **Model-First Development:** The primary focus was on developing and optimizing machine learning models, requiring rapid iteration without database overhead.  
* **Simplicity and Deployment:** JSON files provide simple deployment and version control capabilities.  
* **Performance:** Direct file access provides faster data retrieval for static Quranic text and reciter information.  
* **Machine Learning Integration:** The approach aligns well with ML workflows where models and configurations are stored as files.  
* **Development Efficiency:** Enabled quick prototyping and easy migration between development environments.

  5. ## **Packages and Classes Description** {#packages-and-classes-description}

The implementation utilizes numerous specialized packages and libraries to achieve the required functionality. Table 4.1 provides a comprehensive overview of key packages used in both frontend and backend components, organized by their functional categories and including version specifications for reproducible builds.

*Table ‎4.1: Key packages and libraries used in the Qurrā system implementation*

| Component | Package | Version | Purpose |
| :---: | :---: | :---: | :---: |
| **Backend Core** | Flask | 2.3.0+ | Web framework for API endpoints and server functionality |
|  | joblib | \>=1.0.0 | Model serialization and efficient loading/saving |
|  | numpy | \>=1.20.0 | Numerical computing and array operations |
|  | pandas | \>=1.3.0 | Data manipulation and analysis |
| **Audio Processing** | librosa | \>=0.9.0 | Audio analysis and feature extraction |
|  | pydub | \>=0.25.1 | Audio file format conversion and manipulation |
|  | soundfile | \>=0.10.3 | Audio file I/O operations |
|  | noisereduce | \>=2.0.0 | Audio noise reduction and cleaning |
| **Machine Learning** | scikit-learn | \>=1.0.0 | Traditional ML algorithms and preprocessing |
|  | torch | \>=1.10.0 | PyTorch for deep learning models |
|  | tensorflow | \>=2.8.0 | TensorFlow for neural network operations |
|  | transformers | 4.0+ | Hugging Face transformers for language models |
|  | openai-whisper | 20230918+ | Speech recognition and transcription |
| **Text Processing** | rapidfuzz | 3.0+ | Fuzzy string matching for verse identification |
| **Visualization** | matplotlib | \>=3.5.0 | Plotting and data visualization |
|  | seaborn | \>=0.11.0 | Statistical data visualization |
| **Frontend Core** | react | 18.3.1 | Core React library for UI components |
|  | react-native | 0.76.8 | Cross-platform mobile development framework |
|  | expo | \~52.0.42 | React Native development platform |
|  | expo-router | \~4.0.19 | File-based routing for navigation |
| **Audio (Mobile)** | expo-av | \~15.0.2 | Audio recording and playback functionality |
|  | expo-document-picker | \~13.0.3 | File selection and upload capabilities |
| **UI Components** | @expo/vector-icons | ^14.0.2 | Icon library for mobile interface |
|  | react-native-reanimated | \~3.16.1 | Smooth animations and transitions |
|  | react-native-gesture-handler | \~2.20.2 | Touch gesture handling |
| **Fonts & Typography** | @expo-google-fonts/poppins | ^0.3.0 | Primary font family |
|  | @expo-google-fonts/amiri | ^0.3.0 | Arabic text rendering |
| **HTTP & Networking** | axios | ^1.8.4 | HTTP client for API communication |
| **Development Tools** | typescript | ^5.3.3 | Type safety and enhanced development |
|  | eslint | ^8.57.0 | Code linting and quality assurance |
|  | prettier | ^3.5.3 | Code formatting and style consistency |

As shown in Table 4.1, the technology stack is carefully selected to address the specific requirements of Quranic audio processing and cross-platform mobile development. The backend components focus on robust audio processing capabilities through librosa and pydub, while machine learning functionality is supported by both traditional libraries (scikit-learn) and deep learning frameworks (PyTorch, TensorFlow). The frontend implementation leverages the Expo ecosystem for streamlined React Native development, with specialized packages for Arabic text rendering and audio handling. This comprehensive technology stack ensures reliable performance across all system components while maintaining compatibility and version stability for production deployment.

**Key Classes and Components:**

1. **Backend Classes:**  
   * **ReciterService:** Core business logic for reciter identification, handling audio processing and model inference  
   * **AyahService:** Service layer for verse identification using Whisper transcription and fuzzy matching  
   * **QuranMatcher:** Specialized class for Arabic speech recognition and verse matching algorithms  
   * **AudioProcessor:** Utility class for audio format conversion, noise reduction, and preprocessing  
   * **ModelLoader:** Management class for loading and caching machine learning models efficiently  
2. **Frontend Components:**  
   * **Recorder:** Custom component for audio recording with visual feedback and duration controls  
   * **Uploader:** File upload component with format validation and progress indication  
   * **PredictedReciter:** Result display component showing reciter information and confidence scores  
   * **QuranVerseDisplay:** Specialized component for rendering Arabic text with proper formatting  
   * **NavigationTab:** Custom navigation component maintaining consistent UI across screens

   6. ## **Procedures Description** {#procedures-description}

This section highlights the critical implementation procedures that ensure the system's effectiveness and reliability. Rather than describing all code components, we focus on the most innovative and essential procedures.

**Audio Processing Pipeline**

The audio processing pipeline represents one of the most critical components of the system, handling various audio formats and preparing them for machine learning inference:

 1\. def process\_audio\_file(  
 2\.     file: Union\[str, Path, BinaryIO\],   
 3\.     for\_ayah: bool \= True,  
 4\.     sample\_rate: Optional\[int\] \= None  
 5\. ) \-\> Tuple\[Optional\[np.ndarray\], Union\[int, str\]\]:  
17\.     try:  
18\.         \# Set duration limits based on task  
19\.         min\_duration \= AYAH\_MIN\_DURATION if for\_ayah else RECITER\_MIN\_DURATION  
20\.         max\_duration \= AYAH\_MAX\_DURATION if for\_ayah else RECITER\_MAX\_DURATION  
21\.         target\_sr \= sample\_rate or DEFAULT\_SAMPLE\_RATE  
22\.    
23\.         \# Try librosa first, fallback to pydub for other formats  
24\.         try:  
25\.             y, sr \= librosa.load(file, sr\=None)  
26\.         except Exception as e:  
27\.             logger.debug(f"Direct loading failed, trying pydub: {e}")  
28\.               
29\.             if isinstance(file, (str, Path)):  
30\.                 audio \= AudioSegment.from\_file(file)  
31\.             else:  
32\.                 file.seek(0)  
33\.                 audio \= AudioSegment.from\_file(file)  
34\.               
35\.             \# Convert to normalized numpy array  
36\.             samples \= np.array(audio.get\_array\_of\_samples())  
37\.             if audio.channels \> 1:  
38\.                 samples \= samples.reshape((-1, audio.channels)).mean(axis\=1)  
39\.               
40\.             y \= samples.astype(np.float32) / np.iinfo(samples.dtype).max  
41\.             sr \= audio.frame\_rate  
42\.    
43\.         \# Resample if needed  
44\.         if sr \!= target\_sr:  
45\.             y \= librosa.resample(y, orig\_sr\=sr, target\_sr\=target\_sr)  
46\.             sr \= target\_sr  
47\.    
48\.         \# Validate duration  
49\.         duration \= len(y) / sr  
50\.         if duration \< min\_duration:  
51\.             return None, f"Audio too short. Minimum duration: {min\_duration} seconds"  
52\.           
53\.         if duration \> max\_duration:  
54\.             samples \= int(max\_duration \* sr)  
55\.             y \= y\[:samples\]  
56\.             logger.debug(f"Audio truncated to {max\_duration} seconds")  
57\.    
58\.         return librosa.util.normalize(y), sr  
59\.    
60\.     except Exception as e:  
61\.         logger.error(f"Error processing audio file: {str(e)}")  
62\.         return None, f"Error processing audio file: {str(e)}"  
This procedure implements intelligent audio preprocessing that automatically handles various input formats while maintaining audio quality essential for accurate recognition.

**Dual Model Inference System**

A particularly innovative aspect of the implementation is the dual model inference system that allows comparison between Random Forest and BLSTM approaches:

 1\. def create\_model(model\_type\=None):  
15\.     if model\_type is None:  
16\.         model\_type \= MODEL\_TYPE  
17\.    
18\.     model\_type \= model\_type.lower()  
19\.    
20\.     if model\_type \== 'random\_forest':  
21\.         logger.info("Creating Random Forest model")  
22\.         return RandomForestModel()  
23\.     elif model\_type \== 'blstm':  
24\.         logger.info("Creating Bidirectional LSTM model")  
25\.         return BLSTMModel()  
26\.     else:  
27\.         raise ValueError(f"Unsupported model type: {model\_type}")  
28\.    
29\.    
30\. def load\_model(model\_path\=None):  
41\.     from config import MODEL\_OUTPUT\_DIR  
42\.    
43\.     if model\_path is None:  
44\.         model\_dir \= Path(MODEL\_OUTPUT\_DIR)  
45\.         model\_path \= get\_latest\_model(model\_dir)  
46\.         logger.info(f"Loading latest model: {model\_path}")  
47\.     else:  
48\.         model\_path \= Path(model\_path)  
49\.         if not model\_path.is\_file():  
50\.             raise FileNotFoundError(f"Model file not found: {model\_path}")  
51\.    
52\.     model\_type \= detect\_model\_type\_from\_filename(model\_path)  
53\.    
54\.     \# Try loading with detected type first, then fallback  
55\.     if model\_type \== "blstm":  
56\.         try:  
57\.             return BLSTMModel.load(model\_path)  
58\.         except Exception as e:  
59\.             logger.info("Trying RandomForest as fallback...")  
60\.             return RandomForestModel.load(model\_path)  
61\.     else:  
62\.         try:  
63\.             return RandomForestModel.load(model\_path)  
64\.         except Exception as e:  
65\.             logger.info("Trying BLSTM as fallback...")  
66\.             return BLSTMModel.load(model\_path)  
This factory pattern enables seamless switching between different model architectures while maintaining consistent interfaces for training and inference.

**Arabic Speech Recognition Integration**

The integration of Tarteel's Whisper model for Arabic speech recognition required careful handling of audio preprocessing and transcription:

 1\. def transcribe\_audio(self, audio\_path: str) \-\> str:  
 2\.     """Transcribe audio to text using Whisper model"""  
 3\.     if not hasattr(self, 'model') or not hasattr(self, 'processor'):  
 4\.          raise RuntimeError("QuranMatcher model/processor not properly initialized.")  
 5\.     try:  
 6\.         logging.debug(f"Loading audio file: {audio\_path}")  
 7\.         wav, sr \= torchaudio.load(audio\_path)  
 8\.           
 9\.         \# Ensure mono  
10\.         if wav.shape\[0\] \> 1:  
11\.             logging.debug("Converting stereo audio to mono.")  
12\.             wav \= wav.mean(dim\=0, keepdim\=True)  
13\.    
14\.         \# Check sample rate (should be 16k after conversion)  
15\.         if sr \!= 16000:  
16\.              logging.warning(f"Audio sample rate is {sr}Hz, expected 16000Hz. Resampling...")  
17\.              resampler \= torchaudio.transforms.Resample(orig\_freq\=sr, new\_freq\=16000)  
18\.              wav \= resampler(wav)  
19\.              sr \= 16000  
20\.    
21\.         logging.debug("Extracting features...")  
22\.         inputs \= self.processor(wav.squeeze(0), sampling\_rate\=sr, return\_tensors\='pt')  
23\.         input\_feats \= inputs.input\_features.to(self.device)  
24\.    
25\.         logging.debug("Generating transcription...")  
26\.         \# Use forced\_decoder\_ids for language/task specification  
27\.         forced\_decoder\_ids \= self.processor.get\_decoder\_prompt\_ids(language\="ar", task\="transcribe")  
28\.         pred\_ids \= self.model.generate(input\_feats, forced\_decoder\_ids\=forced\_decoder\_ids)  
29\.         transcription \= self.processor.batch\_decode(pred\_ids, skip\_special\_tokens\=True)\[0\]  
30\.         logging.debug(f"Raw transcription: {transcription}")  
31\.         return transcription  
32\.     except Exception as e:  
33\.         logging.error(f"Transcription failed for {audio\_path}: {e}", exc\_info\=True)  
34\.         raise RuntimeError("Audio transcription failed") from e  
This procedure demonstrates the specialized handling required for Arabic speech recognition, including proper audio preprocessing and language-specific model configuration.

**Fuzzy Matching Algorithm for Verse Identification**

The verse identification system employs sophisticated fuzzy matching to handle partial or imperfect recitations:

 1\. def find\_matches(self, transcription: str, top\_n: int \= 5, score\_threshold: int \= 30) \-\> tuple\[list, str, list\]:  
 2\.     """Find top matching Quran verses via fuzzy matching."""  
 3\.     if not self.all\_verses or not self.normalized\_verses:  
 4\.          logging.error("Cannot find matches: Verse database is empty.")  
 5\.          return \[\], "", \[\]  
 6\.            
 7\.     clean \= self.\_normalize\_text(transcription)  
 8\.     if not clean:  
 9\.         logging.warning("Normalized transcription is empty. Cannot perform matching.")  
10\.         return \[\], clean, \[\]  
11\.    
12\.     \# Get fuzzy scores against precomputed normalized verses  
13\.     try:  
14\.         logging.debug(f"Running fuzzy matching (top\_n={top\_n}, threshold={score\_threshold})...")  
15\.         results \= process.extract(clean, self.normalized\_verses,  
16\.                                   scorer\=fuzz.token\_set\_ratio,  
17\.                                   limit\=top\_n \* 5)  
18\.     except Exception as e:  
19\.         logging.error(f"Fuzzy matching failed: {e}", exc\_info\=True)  
20\.         return \[\], clean, \[\]  
21\.    
22\.     \# Sort by score descending  
23\.     sorted\_res \= sorted(results, key\=lambda x: x\[1\], reverse\=True)  
24\.    
25\.     matches \= \[\]  
26\.     seen \= set()  
27\.     normalized\_matches\_debug \= \[\]  
28\.    
29\.     for choice, score, idx in sorted\_res:  
30\.         \# Threshold check  
31\.         if score \< score\_threshold:  
32\.              logging.debug(f"Score {score:.2f} below threshold {score\_threshold}.")  
33\.              break   
34\.    
35\.         verse \= self.all\_verses\[idx\]  
36\.         key \= (verse\['surah\_num'\], verse\['ayah\_num'\])  
37\.           
38\.         \# Skip duplicates  
39\.         if key in seen:  
40\.             continue  
41\.    
42\.         \# Stop if desired number of unique matches found  
43\.         if len(matches) \>= top\_n:  
44\.             break  
45\.               
46\.         seen.add(key)  
47\.         match\_data \= {  
48\.             'surah\_number': verse\['surah\_num'\],  
49\.             'surah\_name': verse\['surah\_name'\],  
50\.             'surah\_name\_en': verse\['surah\_name\_en'\],  
51\.             'ayah\_number': verse\['ayah\_num'\],   
52\.             'ayah\_text': verse\['original'\],   
53\.             'confidence\_score': float(score),   
54\.             'unicode': ''  
55\.         }  
56\.         matches.append(match\_data)  
57\.    
58\.     logging.debug(f"Found {len(matches)} final unique matches meeting threshold.")  
59\.     return matches, clean, normalized\_matches\_debug  
This algorithm implements intelligent matching that accounts for variations in pronunciation and transcription accuracy while maintaining high precision in verse identification.

**Model Training and Optimization Pipeline**

The model training pipeline implements sophisticated data augmentation and cross-validation for optimal performance:

 1\. def train\_pipeline(model\_type\=None, preprocess\_dir\=None):  
 2\.     """  
 3\.     Run the enhanced training pipeline with improved organization and reporting.  
 4\.     """  
 5\.     try:  
 6\.         total\_start\_time \= time.time()  
 7\.         timestamp \= datetime.now().strftime("%Y%m%d\_%H%M%S")  
 8\.         run\_id \= f"{timestamp}\_train"  
 9\.    
10\.         \# Setup logging  
11\.         logger, log\_file \= setup\_logging(f"training\_{run\_id}")  
12\.         logger.info("Starting enhanced training pipeline")  
13\.    
14\.         \# Load preprocessed data  
15\.         features\_file \= preprocessed\_path / "all\_features.npy"  
16\.         metadata\_file \= preprocessed\_path / "all\_metadata.csv"  
17\.           
18\.         features \= np.load(features\_file)  
19\.         metadata \= pd.read\_csv(metadata\_file)  
20\.         labels \= metadata\['reciter'\].values  
21\.    
22\.         \# Split data  
23\.         X\_train, X\_test, y\_train, y\_test \= train\_test\_split(  
24\.             features, labels, test\_size\=TEST\_SIZE, random\_state\=RANDOM\_STATE, stratify\=labels  
25\.         )  
26\.    
27\.         \# Create and train model  
28\.         model \= create\_model(model\_type)  
29\.         model.train(X\_train, y\_train)  
30\.    
31\.         \# Evaluate model  
32\.         predictions \= model.predict(X\_test)  
33\.         metrics \= model.evaluate(X\_test, y\_test)  
34\.         logger.info(f"Accuracy: {metrics\['accuracy'\]:.4f}")  
35\.    
36\.         \# Save model  
37\.         model\_filename \= model\_dir / f'model\_{model\_type\_str}.joblib'  
38\.         model.save(model\_filename)  
39\.         logger.info(f'Saved model to {model\_filename}')  
40\.    
41\.         return 0, str(model\_filename)  
42\.    
43\.     except Exception as e:  
44\.         logger.error(f"An unexpected error occurred: {str(e)}")  
45\.         return 1, None  
This training pipeline ensures robust model performance through proper data splitting, comprehensive evaluation, and reliability threshold calculation.

**Feature Extraction Pipeline**

The feature extraction system represents a critical component that transforms raw audio signals into meaningful representations for machine learning models:

 1\. def extract\_features(X: np.ndarray, sample\_rate: int) \-\> np.ndarray:  
 2\.     """  
 3\.     Extract audio features for reciter identification.  
 4\.     """  
 5\.     try:  
 6\.         features \= \[\]  
 7\.    
 8\.         \# MFCCs capture the tonal qualities and vocal characteristics  
 9\.         mfccs \= np.mean(librosa.feature.mfcc(y\=X, sr\=sample\_rate, n\_mfcc\=N\_MFCC).T, axis\=0)  
10\.         delta\_mfccs \= np.mean(librosa.feature.delta(mfccs).T, axis\=0)  
11\.         delta2\_mfccs \= np.mean(librosa.feature.delta(mfccs, order\=MFCC\_DELTA\_ORDER).T, axis\=0)  
12\.         features.extend(\[mfccs, delta\_mfccs, delta2\_mfccs\])  
13\.    
14\.         \# Spectral features help identify unique aspects of voice timbre  
15\.         stft \= np.abs(librosa.stft(X))  
16\.         chroma \= np.mean(librosa.feature.chroma\_stft(S\=stft, sr\=sample\_rate, n\_chroma\=N\_CHROMA).T, axis\=0)  
17\.         mel \= np.mean(librosa.feature.melspectrogram(y\=X, sr\=sample\_rate, n\_mels\=N\_MEL\_BANDS).T, axis\=0)  
18\.         contrast \= np.mean(librosa.feature.spectral\_contrast(S\=stft, sr\=sample\_rate).T, axis\=0)  
19\.         rolloff \= np.mean(librosa.feature.spectral\_rolloff(y\=X, sr\=sample\_rate).T, axis\=0)  
20\.         centroid \= np.mean(librosa.feature.spectral\_centroid(y\=X, sr\=sample\_rate).T, axis\=0)  
21\.         features.extend(\[chroma, mel, contrast, rolloff, centroid\])  
22\.    
23\.         \# Additional features for enhanced discriminative power  
24\.         zcr \= np.mean(librosa.feature.zero\_crossing\_rate(X).T, axis\=0)  
25\.         rms \= np.mean(librosa.feature.rms(y\=X).T, axis\=0)  
26\.         tempogram \= np.mean(librosa.feature.tempogram(y\=X, sr\=sample\_rate).T, axis\=0)  
27\.         tonnetz \= np.mean(librosa.feature.tonnetz(y\=librosa.effects.harmonic(X), sr\=sample\_rate).T, axis\=0)  
28\.         features.extend(\[zcr, rms, tempogram, tonnetz\])  
29\.    
30\.         \# Stack everything into one feature vector  
31\.         combined\_features \= np.hstack(\[f.flatten() for f in features\])  
32\.         return combined\_features  
33\.           
34\.     except Exception as e:  
35\.         logger.error(f"Feature extraction failed: {str(e)}")  
36\.         return None  
37\.    
This comprehensive feature extraction pipeline combines multiple audio analysis techniques to capture the unique characteristics of different reciters. The MFCC features form the foundation by capturing vocal tract characteristics, while spectral features provide additional discriminative information about voice timbre and recitation style.

7. ## **Summary** {#summary-3}

This chapter has detailed the comprehensive implementation of the Qurra system, covering all major technical aspects from development tools to critical algorithms. The implementation successfully integrates multiple technologies including React Native for cross-platform mobile development, Python Flask for backend services, and sophisticated machine learning pipelines for audio recognition.

Key implementation achievements include:

* **Technology Integration:** Successful integration of modern mobile development frameworks with advanced machine learning capabilities, creating seamless user experience while maintaining high accuracy in recognition tasks.  
* **Cross-Platform Development:** Implementation of a unified React Native application that provides consistent functionality across iOS and Android platforms, with platform-specific optimizations for audio processing and user interface elements.  
* **Machine Learning Pipeline:** Development of a comprehensive ML pipeline supporting dual model architectures (Random Forest and BLSTM) with automated training, validation, and deployment capabilities.  
* **Arabic Speech Processing:** Successful integration of specialized Arabic speech recognition using Tarteel's Arabic-optimized Whisper model, with custom preprocessing and fuzzy matching algorithms for robust verse identification.  
* **Performance Optimization:** Implementation of efficient audio processing pipelines, model caching mechanisms, and optimized API endpoints ensuring response times under 5 seconds as required.  
* **Data Management:** Development of a flexible file-based data management system optimized for machine learning workflows while maintaining simplicity and portability.  
* **Modular Architecture:** Implementation of a well-structured modular design with clearly separated components for data processing, feature extraction, model management, and API services, enabling easy maintenance and future feature additions.

The implementation demonstrates the successful application of modern software engineering practices to create a practical solution for Quranic audio recognition. The modular architecture and comprehensive testing ensure system reliability while maintaining flexibility for future enhancements. The next chapter will present detailed testing methodologies and results that validate the system's performance and reliability.

**Chapter Five: Testing and Results**

5. # **Testing and Results** {#testing-and-results}

   1. ## **Introduction** {#introduction-5}

This chapter presents the comprehensive testing methodology and results for the Qurra system, demonstrating the effectiveness of the implemented dual recognition capabilities for Quran reciter and verse identification. The testing process validated both technical performance and practical usability across various scenarios and conditions.

The testing approach encompassed multiple levels of validation, from individual component testing to integrated system evaluation. The primary focus was on validating the machine learning models' accuracy, the mobile application's functionality, and the overall system's reliability under different operating conditions.

2. ## **System Testing** {#system-testing}

   1. ### **Unit Testing** {#unit-testing}

Unit testing was conducted for various components of the Qurra system to ensure individual functionality and identify potential issues before integration. 

**Backend Component Testing**

* **Audio Processing Module:** Successfully processed 15 different audio formats with 85% noise reduction effectiveness  
* **Feature Extraction:** Achieved 100% success rate for correct feature vector dimensions with 0.8 seconds average processing time  
* **Model Loading:** Random Forest: 0.3s loading, BLSTM: 1.2s loading with 100% inference reliability

**Frontend Component Testing**

* **Audio Recording:** Consistent 22kHz sample rate across devices with accurate timing controls  
* **File Upload:** Successfully handled MP3, WAV, M4A formats with proper size validation  
* **UI Components:** Correct Arabic text rendering and responsive design across screen sizes

  2. ### **Integration Testing** {#integration-testing}

Integration testing focused on validating the interaction between different system components, particularly the communication between the mobile application and backend services, as well as the coordination between multiple machine learning models.

**API Integration Testing**

* **Mobile-Backend Communication:** The REST API endpoints were tested for proper request handling, response formatting, and error management. Testing included network connectivity scenarios and concurrent user simulations.  
  * **Response Time:** Average API response time of 3.2 seconds for reciter identification  
  * **Success Rate:** 98.5% successful API calls under normal conditions  
  * **Error Handling:** Proper error codes and messages for various failure scenarios  
* **Data Transmission:** Audio file transmission between mobile application and server was validated for integrity and security. Testing confirmed successful file transfer without corruption or security vulnerabilities.

**Model Integration Testing**

* **Dual Model Coordination:** The system's ability to coordinate between Random Forest and BLSTM models was tested to ensure consistent results and proper fallback mechanisms.  
  * **Model Version Compatibility:** Resolved inconsistencies between model serialization formats  
  * **Memory Management:** Optimized model loading to prevent memory overflow during concurrent processing  
  * **Result Synchronization:** Implemented proper result aggregation when using multiple models

    3. ### **Performance Testing** {#performance-testing}

Performance testing evaluated the system's ability to meet specified requirements under various load conditions and operational scenarios. The testing focused on four critical metrics that directly impact user experience and system reliability.

Table 5.1 presents the performance evaluation results comparing target requirements against achieved metrics. The processing time metric measures the end-to-end duration from audio input submission to result delivery, which is crucial for user satisfaction. Reciter accuracy represents the overall success rate in correctly identifying known reciters from the training dataset. The reliability rate indicates the percentage of predictions where the system demonstrates high confidence in its results, helping users understand when to trust the identification. API success rate measures the system's ability to handle requests without errors under normal operating conditions.

*Table ‎5.1: Performance Testing Results*

| Metric | Target | Achieved | Status |
| ----- | ----- | ----- | ----- |
| Processing Time | \<5 seconds | 3.2 seconds avg | ✓ PASS |
| Reciter Accuracy | \>90% | 93.57% overall | ✓ PASS |
| Reliability Rate | \>85% | 85.38% | ✓ PASS |
| API Success Rate | \>95% | 98.5% | ✓ PASS |

As shown in Table 5.1, all performance targets were successfully met or exceeded, with processing time achieving a 36% improvement over the maximum requirement and reciter accuracy surpassing the target by 3.57 percentage points.

4. ### **Test Cases** {#test-cases}

Critical functionality was validated through specific test scenarios designed to evaluate the system's performance under various real-world conditions. The test cases were structured to cover both optimal operating conditions and challenging scenarios that users might encounter in practical usage.

Table 5.2 presents comprehensive test case results, encompassing five key testing scenarios. Test Case TC-001 validates the system's performance under ideal conditions with high-quality audio inputs, representing the baseline functionality expected from the system. TC-002 examines the system's ability to handle unknown reciters, which is crucial for preventing false positive identifications when users submit audio from reciters not included in the training dataset. TC-003 evaluates the Ayah identification capabilities, testing the integration of Arabic speech recognition and fuzzy matching algorithms. TC-004 ensures consistent user experience across different mobile platforms, validating the cross-platform architecture's effectiveness. TC-005 tests the system's robustness against audio quality degradation, simulating real-world recording conditions with background noise.

*Table ‎5.2: Test Case Results and Validation*

| Test Case | Description | Expected Result | Actual Result | Status |
| :---- | :---- | :---- | :---- | :---- |
| TC-001 | Reciter ID with high-quality audio | \>90% confidence | 95-100% confidence | PASS |
| TC-002 | Non-training reciter handling | Low confidence/rejection | 75.44% accuracy (24.56% false positive) | PARTIAL |
| TC-003 | Ayah identification | \>70% matching | 78% accuracy for clear audio | PASS |
| TC-004 | Cross-platform consistency | Uniform performance | Consistent across iOS/Android | PASS |
| TC-005 | Noise handling | \>80% with moderate noise | 85% accuracy maintained | PASS |

Table 5.2 reveals that four out of five test cases achieved full success, with TC-002 receiving a partial pass due to higher-than-expected false positive rates for non-training reciters. This result indicates an area requiring future enhancement in unknown reciter detection capabilities.

3. ## **Discussion** {#discussion}

Comprehensive testing of the Qurra system reveals significant achievements while identifying areas for future improvement. The results demonstrate that the primary objectives of accurate reciter identification have been successfully met, with the system achieving 93.57% overall accuracy and 94.86% accuracy for reliable predictions.

**Key Achievements**

* **High Accuracy for Training Reciters:** The system demonstrates exceptional performance for reciters included in the training dataset, with some reciters achieving 100% accuracy. This validates the effectiveness of the MFCC feature extraction and Random Forest classification approach.   
* **Reliable Processing Speed:** With an average response time of 3.2 seconds, the system successfully meets the requirement of processing requests in under 5 seconds, providing users with responsive experience.   
* **Cross-Platform Consistency:** The React Native implementation successfully delivers consistent functionality across iOS and Android platforms, with minimal performance differences between platforms.   
* **Effective Audio Processing:** The comprehensive audio preprocessing pipeline handles various input formats effectively, with noise reduction capabilities improving recognition accuracy by approximately 15% in noisy conditions.

**Challenges and Limitations**

* **Non-Training Reciter Recognition:** The system's 24.56% false positive rate for non-training reciters represents a significant limitation, indicating the current model lacks robust mechanisms for detecting unknown reciters.   
* **Dataset Size Limitations:** With only 5 reciters in the training dataset, the system's practical applicability is limited. Expanding to handle a broader range of reciters would require significant dataset expansion and retraining.   
* **Noise Sensitivity:** While the system handles moderate noise levels adequately, performance degrades significantly with heavy noise (SNR \<10dB), dropping to 78% accuracy.   
* **Ayah Identification Accuracy:** The 70% success rate for partial clips indicates room for improvement in the Arabic speech recognition and fuzzy matching algorithms.

  4. ## **Summary** {#summary-4}

This chapter has presented comprehensive testing results for the Qurra system, demonstrating successful achievement of the primary project objectives. The system achieves 93.57% overall accuracy in reciter identification, with processing times well within the specified requirements and consistent cross-platform functionality.

**Key Testing Outcomes:**

* **Performance Validation:** The system meets or exceeds all primary performance requirements, including accuracy targets (\>90%) and processing speed requirements (\<5 seconds). The reliability rate of 85.38% indicates that the system provides dependable results for most use cases.  
* **Functional Verification:** Comprehensive testing across multiple scenarios validates that all major system functions operate as designed, from audio input processing to result display and cross-platform compatibility.  
* **Model Effectiveness:** The dual recognition approach proves effective, with the Random Forest model demonstrating superior performance compared to baseline approaches, while the integrated Whisper model provides valuable Ayah identification capabilities.  
* **Integration Success**: The complex integration of mobile applications, backend services, and multiple machine learning models operates cohesively, with successful API communication and proper error handling throughout the system.  
* **Areas for Enhancement:** Testing identified specific areas for future improvement, particularly in handling non-training reciters and improving noise robustness, providing clear direction for continued development.

The testing results confirm that the Qurra system successfully addresses the identified problem of Quran reciter and verse identification, providing a practical tool for Muslims worldwide while establishing a strong foundation for future enhancements and expanded capabilities. The next chapter will conclude the report with an assessment of achievements and outline future development directions.

**Chapter Six: Conclusion and Future Work**

6. # **Conclusion and Future Work** {#conclusion-and-future-work}

   1. ## **Conclusion** {#conclusion}

The Qurrā project successfully delivers the first comprehensive mobile application that combines Quran reciter identification with verse recognition capabilities. This research demonstrates the practical application of artificial intelligence to Islamic educational technology, addressing a critical gap in serving the global Muslim community.

**Major Achievements**

* **Technical Success**: The system achieved 93.57% accuracy in reciter identification using a dual-model approach (Random Forest and BLSTM) while processing requests within 3.2 seconds, exceeding the 5-second requirement. The integration of Tarteel's Whisper model enables Ayah identification with 70-90% accuracy for complete verses.  
* **Innovation Contributions**: The project introduces several technical innovations:  
  * First dual recognition architecture combining reciter and verse identification

* Specialized audio preprocessing pipeline optimized for Quranic recitation

* Cross-platform React Native implementation with sophisticated Flask backend

* Confidence scoring mechanisms ensuring trustworthy AI in religious contexts

* **Impact Realization**: Within Saudi Arabia's Vision 2030 framework, the project demonstrates successful application of advanced technologies to cultural heritage preservation. Globally, it serves approximately 1.8 billion Muslims, particularly those in non-Arabic speaking countries with limited access to traditional Quranic education.

**Challenges Overcome**

The development process addressed significant technical challenges including model integration complexity, Arabic speech processing requirements, and cross-platform deployment optimization. The 24.56% false positive rate for non-training reciters represents a current limitation but provides a clear direction for future enhancement.

**Research Validation**

Comprehensive testing validates the achievement of all primary objectives. The system's ability to process various audio formats while maintaining high accuracy demonstrates practical viability for real-world deployment, establishing a new standard for Islamic educational technology.

2. ## **Future Work** {#future-work}

**Immediate Enhancements**

* **Dataset Expansion**: Scale from 5 to 20-50 reciters while maintaining accuracy. Implement different Qirāʾāt recognition capabilities to serve diverse recitation traditions.  
* **Reliability Improvement**: Develop anomaly detection mechanisms to address the current limitation in handling unknown reciters, reducing false positive rates through confidence-based rejection strategies.  
* **Offline Functionality**: Implement model compression and edge computing techniques to enable offline recognition capabilities, crucial for users in areas with limited connectivity.

**Advanced Features**

* **Multilingual Support**: Expand beyond Arabic and English to include Urdu, Turkish, Indonesian, and French interfaces, reflecting major Muslim populations worldwide.  
* **Educational Integration**: Develop APIs for integration with madrasas and Islamic educational platforms, including educator-specific features like progress tracking and assessment tools.  
* **Personalization**: Implement user analytics, personalized recommendations, and adaptive learning systems to enhance educational value.

**Technical Infrastructure**

* **Architecture Modernization**: Transition to cloud-native microservices architecture and implementation of relational database systems for enhanced scalability and data integrity.  
* **Research Initiatives**: Establish partnerships with Islamic universities to advance Quranic audio processing research and develop open-source components for broader community benefit.

The future development of Qurra represents not just technical enhancement but an opportunity to set new standards for respectful, effective application of artificial intelligence in serving religious and cultural communities worldwide. Through careful planning, community engagement, and continued innovation, the system can evolve to serve the diverse and growing needs of the global Muslim community while contributing to the broader fields of Arabic language processing and Islamic educational technology.

# **References** {#references}

\[1\]	M. Al-Ayyoub, N. Alhuda, M. A. Damer, I. Hmeidi, and A. Damer, “Using Deep Learning for Automatically Determining Correct Application of Basic Quranic Recitation Rules,” 2018\. \[Online\]. Available: https://www.researchgate.net/publication/326010238

\[2\]	“Tarteel AI Quran Memorization.” Accessed: Jan. 23, 2025\. \[Online\]. Available: https://www.tarteel.ai/

\[3\]	“GitHub \- salsowelim/nadyy\_dataset.” Accessed: Jan. 23, 2025\. \[Online\]. Available: https://github.com/salsowelim/nadyy\_dataset

\[4\]	B. Yousfi and A. M. Zeki, “Holy Qur’an speech recognition system distinguishing the type of recitation,” *Proceedings \- CSIT 2016: 2016 7th International Conference on Computer Science and Information Technology*, Aug. 2016, doi: 10.1109/CSIT.2016.7549454.

\[5\]	K. M. O. Nahar, R. M. Al-Khatib, M. A. Al-Shannaq, and M. M. Barhoush, “An efficient holy quran recitation recognizer based on SVM learning model,” *Jordanian Journal of Computers and Information Technology*, vol. 6, no. 4, pp. 392–414, Dec. 2020, doi: 10.5455/JJCIT.71-1593380662.

\[6\]	K. Cho, B. van Merriënboer, D. Bahdanau, and Y. Bengio, “On the properties of neural machine translation: Encoder–decoder approaches,” *Proceedings of SSST 2014 \- 8th Workshop on Syntax, Semantics and Structure in Statistical Translation*, pp. 103–111, Oct. 2014, doi: 10.3115/V1/W14-4012.

\[7\]	R. Lu and Z. Duan, “Bidirectional GRU for Sound Event Detection,” Munich, Nov. 2017\.

\[8\]	A. Qayyum, S. Latif, and J. Qadir, “Quran Reciter Identification: A Deep Learning Approach,” *Proceedings of the 2018 7th International Conference on Computer and Communication Engineering, ICCCE 2018*, pp. 492–497, Nov. 2018, doi: 10.1109/ICCCE.2018.8539336.

\[9\]	A. Al Harere and K. Al Jallad, “Quran Recitation Recognition using End-to-End Deep Learning,” May 2023, Accessed: Jan. 23, 2025\. \[Online\]. Available: https://arxiv.org/abs/2305.07034v1

\[10\]	M. A. Hussaini and R. W. Aldhaheri, “An Automatic Qari Recognition System,” *Proceedings \- 2012 International Conference on Advanced Computer Science Applications and Technologies, ACSAT 2012*, pp. 524–528, 2012, doi: 10.1109/ACSAT.2012.48.

\[11\]	H. Eyadah and A. Odaibat, “A Forward-Looking Vision to Employ Artificial Intelligence to Preserve Cultural Heritage,” *Humanities and Social Sciences*, vol. 12, no. 5, pp. 109–114, Sep. 2024, doi: 10.11648/J.HSS.20241205.12.

\[12\]	N. R. Aljohani, M. A. Aslam, A. O. Khadidos, and S. U. Hassan, “A Methodological Framework to Predict Future Market Needs for Sustainable Skills Management Using AI and Big Data Technologies,” *Applied Sciences*, vol. 12, no. 14, Jul. 2022, doi: 10.3390/APP12146898.

\[13\]	A. Rahman, M. M. Kabir, M. F. Mridha, M. Alatiyyah, H. F. Alhasson, and S. S. Alharbi, “Arabic Speech Recognition: Advancement and Challenges,” *IEEE Access*, vol. 12, pp. 39689–39716, 2024, doi: 10.1109/ACCESS.2024.3376237.

\[14\]	M. Elsayess and S. Hamada, “Using Mobile Application in Teaching Correct Recitation of the Holy Quran,” 2013, doi: 10.1109/NOORIC.2013.97.

\[15\]	K. A. Ahmad *et al.*, “Mobile Learning of Islamic Studies: A Comprehensive Review,” *Journal of Advanced Research in Applied Sciences and Engineering Technology*, vol. 48, no. 2, pp. 211–224, Jun. 2025, doi: 10.37934/ARASET.48.2.211224.

\[16\]	S. H. C. Hassan, S. N. W. Shamsuddin, and N. H. Yusof, “Towards Designing a Framework for Adaptive Gamification Learning Analytics in Quranic Memorisation,” *Pertanika J Sci Technol*, vol. 31, no. 1, pp. 257–278, 2023, doi: 10.47836/PJST.31.1.16.

\[17\]	N. H. A. H. B. Zahari, S. N. B. B. S. Bidin, and S. N. B. W. Syamsuddin, “Development of Al-Quran Android Application from Year 2013 To 2016: The Highlight,” *The International Journal of Academic Research in Business and Social Sciences*, vol. 7, no. 6, pp. 183–195, Jun. 2017, doi: 10.6007/IJARBSS/V7-I6/2954.

\[18\]	SamaraGhassan, Al-DaoudEssam, SwerkiNael, and Alzu’biDalia, “The Recognition of Holy Qur’an Reciters Using the MFCCs’ Technique and Deep Learning,” *Advances in Multimedia*, vol. 2023, pp. 1–14, Mar. 2023, doi: 10.1155/2023/2642558.

\[19\]	M. Tall, T. I. Diop, N. Fatou Ngom, E. Hadj, and A. Thiam, “Deep learning for Quranic reciter recognition and audio content identification.”

\[20\]	“Sunan an-Nasa’i 1015 \- The Book of the Commencement of the Prayer \- كتاب الافتتاح \- Sunnah.com \- Sayings and Teachings of Prophet Muhammad (صلى الله عليه و سلم).” Accessed: Jan. 24, 2025\. \[Online\]. Available: https://sunnah.com/nasai:1015

\[21\]	N. Alias, K. A. Mohamad, Z. Mohamed Nor, and N. H. Mat Saad, “Sanad Qiraat dan Hadis: Analisa Terhadap Proses Pengijazahan,” *Maʿālim al-Qurʾān wa al-Sunnah*, vol. 14, no. 1, pp. 109–120, Jun. 2018, doi: 10.33102/JMQS.V14I1.122.

\[22\]	M. Ibrahim and E.-K. Ghembaza, “A Knowledge-based System for Extracting Combined and Individual Quranic Recitations,” 2019\.

\[23\]	M. Lataifeh and A. Elnagar, “Ar-DAD: Arabic diversified audio dataset,” *Data Brief*, vol. 33, p. 106503, Dec. 2020, doi: 10.1016/J.DIB.2020.106503. 

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,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>

[image3]: <data:image/png;base64,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>

[image4]: <data:image/png;base64,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>

[image5]: <data:image/png;base64,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>

[image6]: <data:image/png;base64,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>

[image7]: <data:image/png;base64,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>

[image8]: <data:image/png;base64,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>

[image9]: <data:image/png;base64,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>

[image10]: <data:image/png;base64,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>

[image11]: <data:image/png;base64,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>

[image12]: <data:image/png;base64,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>