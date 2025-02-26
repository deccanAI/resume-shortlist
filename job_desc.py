def classify_exp(yoe):
    """Classify years of experience into categories."""
    if yoe < 3:
        return "jr"
    elif yoe < 6:
        return "mid"
    else:
        return "senior"
    
ml_engineer = {
    "jr": """Machine Learning Engineer (0-2 Years Experience) – Job Description
The selected candidate will work on developing, deploying, and maintaining machine learning models, focusing on model training, evaluation, and basic MLOps practices. The role involves collaborating with data scientists and engineers to ensure models are effectively integrated into production systems.
Responsibilities
Assist in data preprocessing, feature engineering, and exploratory data analysis (EDA) to prepare datasets for training.
Develop and train machine learning models using frameworks like TensorFlow, PyTorch, or Scikit-learn.
Implement and fine-tune model evaluation metrics to ensure performance meets business requirements.
Deploy ML models into production using Flask, FastAPI, or cloud-based ML services (AWS SageMaker, GCP Vertex AI, Azure ML).
Work with basic CI/CD pipelines for automating model training and deployment.
Write and maintain clean, efficient, and well-documented code in Python.
Debug and optimize models for better efficiency and performance.
Collaborate with data engineers and software developers to integrate ML models into production systems.
Assist in monitoring model performance and retraining models when necessary.
Learn and apply MLOps best practices for versioning, logging, and tracking experiments using MLflow or Weights & Biases.
Required Skills
Programming: Proficiency in Python; familiarity with libraries like Pandas, NumPy, Scikit-learn, TensorFlow, or PyTorch.
Machine Learning: Understanding of supervised and unsupervised learning, feature engineering, and hyperparameter tuning.
Data Handling: Experience working with structured and unstructured data, including basic SQL and data visualization.
Model Deployment: Exposure to deploying models using Flask, FastAPI, or cloud-based ML services.
Version Control & CI/CD: Familiarity with Git and basic CI/CD concepts.
MLOps Basics: Understanding of ML model versioning and tracking tools like MLflow or DVC.
Cloud Basics: Some exposure to AWS, GCP, or Azure ML tools.
Problem-Solving: Strong analytical and problem-solving skills.
Nice to Have
Experience with Docker and Kubernetes for containerizing ML models.
Familiarity with Big Data processing frameworks like Spark or Dask.
Basic knowledge of deep learning techniques.
Some exposure to model monitoring tools like Prometheus or Evidently AI.
""",
    "mid": """Machine Learning Engineer (3-5 Years Experience) – Job Description
The selected candidate will focus on building, deploying, and optimizing machine learning models for production environments. Responsibilities include developing ML pipelines, integrating models with cloud infrastructure, monitoring performance, and automating model retraining workflows.
Responsibilities
Develop and optimize machine learning models using Python, TensorFlow, PyTorch, or Scikit-learn.
Implement data preprocessing, feature engineering, and model evaluation techniques.
Design and maintain end-to-end ML pipelines using MLflow, TFX, or Kubeflow.
Deploy machine learning models into production using Docker, Kubernetes, or serverless architectures.
Automate model training, versioning, and deployment workflows with CI/CD tools like GitHub Actions, GitLab CI/CD, or Jenkins.
Work with large-scale datasets, handling structured and unstructured data efficiently.
Optimize data pipelines using Apache Spark, Dask, or Apache Airflow.
Ensure model monitoring and performance tracking using Prometheus, Grafana, or MLflow.
Implement basic model observability and drift detection techniques using Evidently AI or WhyLabs.
Work with cloud-based ML platforms such as AWS SageMaker, GCP Vertex AI, or Azure ML for scalable deployment.
Ensure security and compliance of ML workflows, handling model governance and explainability with tools like SHAP or LIME.
Required Skills
Strong understanding of machine learning concepts, including supervised and unsupervised learning.
Hands-on experience with ML frameworks like TensorFlow, PyTorch, or Scikit-learn.
Experience in building and deploying ML pipelines using MLflow, TFX, or Kubeflow.
Knowledge of cloud-based ML services such as AWS SageMaker, GCP Vertex AI, or Azure ML.
Proficiency in programming with Python and experience with data manipulation libraries like Pandas and NumPy.
Familiarity with containerization and orchestration tools such as Docker and Kubernetes.
Experience with MLOps practices, including CI/CD pipelines and automated model retraining.
Nice to Have
Experience with A/B testing and hyperparameter tuning for model optimization.
Familiarity with feature stores like Feast or AWS Feature Store.
Exposure to responsible AI techniques, explainability, and bias detection tools.
""", 
    "sr": """Machine Learning Engineer (5+ Years Experience) – Job Description
The selected candidate will focus on designing, deploying, and optimizing end-to-end machine learning solutions at scale. Responsibilities include developing and fine-tuning models, optimizing data pipelines, deploying ML models in production, ensuring model reliability and observability, and integrating with cloud-based infrastructures.
Responsibilities
Design, develop, and optimize machine learning models for real-world applications.
Implement feature engineering techniques to improve model accuracy and efficiency.
Leverage deep learning frameworks such as TensorFlow and PyTorch for complex AI applications.
Build and automate scalable ML pipelines using Kubeflow, MLflow, or TFX.
Deploy and manage ML models in production using Docker, Kubernetes, and serverless architectures.
Ensure model versioning, logging, and monitoring for production stability.
Work with large-scale structured and unstructured data from diverse sources.
Develop and maintain feature stores using Feast, AWS SageMaker Feature Store, or Databricks Feature Store.
Optimize data preprocessing pipelines using Spark, Dask, or Ray.
Implement CI/CD pipelines for ML workflows using GitHub Actions, GitLab CI/CD, or Jenkins.
Automate model retraining, monitoring, and performance evaluation.
Ensure reproducibility with ML experiment tracking tools like Weights & Biases or MLflow.
Deploy real-time model monitoring for drift detection using Evidently AI, WhyLabs, or Arize AI.
Conduct performance analysis using Prometheus, Grafana, and OpenTelemetry.
Set up automated alerting for model degradation and bias detection.
Implement security best practices for ML models, including data encryption and access control.
Ensure compliance with regulations such as GDPR, CCPA, and HIPAA.
Develop explainable AI models using SHAP, LIME, or Captum.
Required Skills
Proficiency in machine learning and deep learning, including supervised, unsupervised, and reinforcement learning.
Experience with data engineering and processing tools such as Spark, Dask, Kafka, and Airflow.
Hands-on expertise in MLOps and automation, including CI/CD for ML, Docker, Kubernetes, Kubeflow, MLflow, and TFX.
Strong knowledge of model deployment and scaling with cloud platforms such as AWS SageMaker, GCP Vertex AI, and Azure ML.
Experience in observability and model monitoring using Prometheus, Grafana, Evidently AI, and OpenTelemetry.
Understanding of security, governance, and responsible AI, including model explainability, compliance, and privacy-preserving ML techniques.
Nice to Have
Experience with A/B testing, Bayesian optimization, and hyperparameter tuning.
Knowledge of multi-cloud ML deployments using AWS, GCP, and Azure.
Familiarity with generative AI, LLM fine-tuning, and vector databases such as FAISS, Weaviate, and Pinecone.
"""}

mlops = {
    "jr": """MLOps Engineer (0-2 Years Experience) – Job Description
The selected candidate will be responsible for streamlining the deployment, monitoring, and maintenance of machine learning models in production. The role involves working with data scientists and engineers to ensure smooth integration, automation, and scalability of ML workflows using MLOps best practices.
Responsibilities
Assist in building and maintaining CI/CD pipelines for automating model training, testing, and deployment.
Work on containerizing ML models using Docker and orchestrating them with Kubernetes.
Implement model versioning, tracking, and logging using MLflow, DVC, or similar tools.
Deploy ML models into production environments using cloud services like AWS SageMaker, GCP Vertex AI, or Azure ML.
Monitor model performance and automate retraining workflows to ensure continuous improvement.
Work with data engineers to optimize data pipelines for ML model training and inference.
Ensure scalability, reliability, and security of ML deployments.
Assist in setting up monitoring and alerting for model performance using tools like Prometheus, Grafana, or Evidently AI.
Maintain documentation and best practices for ML model deployment and lifecycle management.
Required Skills
Programming: Proficiency in Python and Bash scripting.
CI/CD & Automation: Basic knowledge of Git, Jenkins, GitHub Actions, or similar CI/CD tools.
Containerization & Orchestration: Experience with Docker and Kubernetes.
Model Deployment: Exposure to deploying ML models using Flask, FastAPI, or cloud-based ML services.
MLOps Fundamentals: Understanding of model versioning, tracking, and logging using MLflow or DVC.
Cloud Computing: Some experience with AWS, GCP, or Azure for ML workloads.
Monitoring & Observability: Basic knowledge of Prometheus, Grafana, or similar monitoring tools.
Problem-Solving: Strong analytical and troubleshooting skills.
Nice to Have
Experience with Terraform or CloudFormation for infrastructure as code.
Familiarity with feature stores like Feast.
Exposure to streaming data pipelines using Kafka or Apache Airflow.
Knowledge of security best practices in ML model deployment.
""",
    "mid": """MLOps Engineer (3-5 Years Experience) – Job Description
The selected candidate will be responsible for designing, deploying, and maintaining scalable machine learning pipelines and infrastructure. The role involves implementing MLOps best practices, ensuring model reliability, automating workflows, and optimizing production ML systems.
Responsibilities
Design and implement scalable end-to-end ML pipelines using tools like MLflow, TFX, or Kubeflow.
Automate model training, testing, and deployment workflows with CI/CD pipelines (GitHub Actions, Jenkins, GitLab CI/CD).
Deploy ML models in production using Docker, Kubernetes, or serverless architectures.
Monitor and manage model performance, drift detection, and observability using Evidently AI, Prometheus, Grafana, or WhyLabs.
Work with large-scale data pipelines, optimizing ETL processes using Apache Spark, Dask, or Apache Airflow.
Implement model versioning, tracking, and governance with MLflow, DVC, or Feature Stores (Feast, AWS Feature Store).
Ensure security, compliance, and explainability of ML models using SHAP, LIME, or model governance frameworks.
Optimize ML infrastructure for scalability, efficiency, and cost-effectiveness in AWS, GCP, or Azure.
Collaborate with data engineers, ML engineers, and DevOps teams to streamline production ML workflows.
Troubleshoot performance bottlenecks, infrastructure failures, and deployment issues in ML systems.
Required Skills
Strong experience in MLOps, model deployment, and production ML pipelines.
Proficiency in Python, Bash scripting, and ML libraries like TensorFlow, PyTorch, or Scikit-learn.
Experience with CI/CD automation, containerization (Docker), and orchestration (Kubernetes, Terraform).
Hands-on experience with cloud ML platforms like AWS SageMaker, GCP Vertex AI, or Azure ML.
Expertise in monitoring and logging ML models using Evidently AI, Prometheus, or MLflow.
Familiarity with data engineering tools such as Apache Spark, Kafka, or Airflow.
Strong understanding of feature engineering, model retraining, and drift detection techniques.
Experience with infrastructure as code (Terraform, CloudFormation) for ML environments.
Nice to Have
Exposure to A/B testing and model validation techniques.
Familiarity with feature stores (Feast, AWS Feature Store) for managing ML features.
Knowledge of responsible AI practices, explainability, and fairness in ML.
Experience with serverless ML deployment (AWS Lambda, Google Cloud Functions).
""",
    "sr": """MLOps Engineer (5+ Years Experience) – Job Description
The selected candidate will be responsible for architecting, automating, and optimizing enterprise-scale ML workflows and infrastructure. This role requires expertise in MLOps best practices, model lifecycle management, cloud-based deployments, and scalable data pipelines. The ideal candidate will drive efficiency, reliability, and automation in machine learning systems, ensuring seamless integration with business applications.
Responsibilities
Architect and optimize large-scale ML infrastructure using Kubeflow, MLflow, TFX, or SageMaker Pipelines.
Design and implement CI/CD pipelines for ML model training, testing, and deployment using GitHub Actions, Jenkins, or GitLab CI/CD.
Automate ML workflows, including data ingestion, feature engineering, model retraining, and deployment.
Deploy and scale ML models in production using Docker, Kubernetes, Airflow, and serverless architectures.
Establish model observability and monitoring frameworks using Evidently AI, Prometheus, Grafana, or WhyLabs to track model drift and data quality.
Ensure high availability and performance of ML systems through distributed computing and cloud-native architectures (AWS, GCP, Azure).
Manage model governance, security, and compliance, implementing model explainability tools (SHAP, LIME) and bias detection frameworks.
Optimize cost and resource allocation for ML workloads using Kubernetes autoscaling, Spot Instances, and serverless ML solutions.
Collaborate with data engineers, DevOps teams, and business stakeholders to streamline end-to-end ML operations.
Lead and mentor junior MLOps engineers, ensuring best practices in ML model deployment and infrastructure management.
Required Skills
Deep expertise in MLOps, ML model deployment, and production-scale ML systems.
Strong programming skills in Python, Bash scripting, and proficiency in TensorFlow, PyTorch, or Scikit-learn.
Advanced knowledge of CI/CD pipelines, infrastructure as code (Terraform, CloudFormation), and cloud-based ML platforms (AWS SageMaker, GCP Vertex AI, Azure ML).
Hands-on experience with Kubernetes, including scalability, load balancing, and autoscaling for ML workloads.
Expertise in monitoring and logging ML models with tools like Evidently AI, MLflow, Prometheus, Grafana.
Experience with feature stores (Feast, AWS Feature Store) and data versioning tools (DVC, Delta Lake).
Strong understanding of security and compliance in ML environments, including IAM roles, model explainability, and ethical AI practices.
Proficiency in distributed data processing using Apache Spark, Kafka, or Dask for large-scale ML pipelines.
Nice to Have
Experience with A/B testing, canary deployments, and shadow testing for ML models.
Familiarity with serverless ML architectures (AWS Lambda, Google Cloud Functions, Azure Functions).
Knowledge of graph-based ML pipelines and real-time model inference.
Exposure to federated learning and privacy-preserving ML techniques.
Contributions to open-source MLOps tools or frameworks.
"""
}

nlp_engineer = {
    "jr": """NLP Engineer (0-2 Years Experience) – Job Description
The selected candidate will be responsible for developing, fine-tuning, and deploying Natural Language Processing (NLP) models. The role involves working with text data, implementing state-of-the-art NLP techniques, and integrating models into production systems.
Responsibilities
Assist in preprocessing and cleaning text data, including tokenization, stemming, lemmatization, and stopword removal.
Develop and fine-tune NLP models using libraries such as Hugging Face Transformers, NLTK, SpaCy, or TensorFlow/Keras.
Implement text vectorization techniques such as TF-IDF, Word2Vec, FastText, or BERT embeddings.
Train and evaluate NLP models for tasks like text classification, named entity recognition (NER), sentiment analysis, and text summarization.
Deploy NLP models as APIs using Flask or FastAPI.
Optimize NLP pipelines for efficiency and scalability in production environments.
Work with large-scale text datasets, performing exploratory data analysis (EDA) to extract insights.
Collaborate with data engineers to integrate NLP solutions into existing data pipelines.
Implement model evaluation metrics such as precision, recall, F1-score, and perplexity for language models.
Maintain clear documentation for models, APIs, and data preprocessing techniques.
Required Skills
Programming: Proficiency in Python with experience in libraries like NumPy, Pandas, and Scikit-learn.
NLP Fundamentals: Understanding of tokenization, word embeddings, POS tagging, and named entity recognition.
Model Development: Hands-on experience with NLP frameworks such as Hugging Face, SpaCy, NLTK, or TensorFlow/Keras for text processing tasks.
Text Preprocessing: Ability to clean and prepare text data for NLP applications.
Model Deployment: Exposure to deploying NLP models as REST APIs using Flask or FastAPI.
Version Control & CI/CD: Familiarity with Git and basic CI/CD concepts.
Cloud Basics: Some exposure to NLP services in AWS, GCP, or Azure.
Problem-Solving: Strong analytical skills with the ability to troubleshoot NLP pipeline issues.
Nice to Have
Experience with transformer-based models such as BERT, GPT, or T5.
Knowledge of information retrieval techniques and search algorithms.
Familiarity with large-scale data processing using Apache Spark or Dask.
Understanding of deploying models using Docker and Kubernetes.
Exposure to vector databases and retrieval-augmented generation (RAG) techniques.
""",
    "mid": """NLP Engineer (3-5 Years Experience) – Job Description
The selected candidate will be responsible for designing, developing, and optimizing Natural Language Processing (NLP) models for real-world applications. The role involves working with large-scale text data, implementing state-of-the-art NLP techniques, deploying models in production, and optimizing performance for scalability and efficiency.
Responsibilities
Design and implement NLP models for tasks such as text classification, named entity recognition (NER), sentiment analysis, machine translation, and question answering.
Preprocess and clean text data, including tokenization, lemmatization, stemming, and entity recognition.
Develop and fine-tune transformer-based models such as BERT, GPT, T5, or LLaMA using Hugging Face Transformers or TensorFlow/Keras.
Optimize NLP pipelines for low-latency inference and efficient serving in production environments.
Deploy NLP models using Flask, FastAPI, or cloud-based services such as AWS SageMaker, GCP Vertex AI, or Azure ML.
Implement model evaluation metrics such as BLEU, ROUGE, perplexity, precision, recall, and F1-score.
Build and maintain scalable NLP data pipelines using Apache Spark, Dask, or Airflow.
Work with vector databases and retrieval-augmented generation (RAG) techniques for NLP applications.
Implement continuous monitoring, logging, and retraining strategies to improve model performance.
Collaborate with software engineers, data scientists, and product teams to integrate NLP solutions into business applications.
Required Skills
Programming: Strong proficiency in Python with experience in libraries like NumPy, Pandas, Scikit-learn, and PyTorch/TensorFlow.
NLP Techniques: In-depth understanding of tokenization, word embeddings, attention mechanisms, and deep learning-based NLP models.
Model Development: Hands-on experience in training and fine-tuning transformer-based models (BERT, GPT, T5, etc.).
Model Deployment: Experience deploying NLP models using REST APIs (Flask, FastAPI) or cloud-based ML platforms.
Scalability & Performance: Experience optimizing NLP models for production, including quantization and pruning techniques.
Data Engineering: Experience working with large-scale structured and unstructured text datasets, including knowledge of SQL and NoSQL databases.
MLOps & CI/CD: Familiarity with model versioning, experiment tracking (MLflow, Weights & Biases), and CI/CD pipelines.
Cloud & Infrastructure: Experience working with cloud-based NLP solutions on AWS, GCP, or Azure.
Nice to Have
Experience with knowledge graphs and semantic search.
Exposure to information retrieval techniques and large-scale search algorithms.
Understanding of deploying NLP models using Kubernetes and serverless architectures.
Experience with prompt engineering and fine-tuning large language models (LLMs).
Familiarity with vector databases like FAISS, Pinecone, or Weaviate for NLP applications.
""",
    "sr": """NLP Engineer (5+ Years Experience) – Job Description
The selected candidate will lead the development, deployment, and optimization of advanced Natural Language Processing (NLP) models for large-scale applications. This role involves designing scalable NLP architectures, fine-tuning large language models (LLMs), integrating NLP systems into production environments, and implementing cutting-edge techniques for improving model performance and efficiency.
Responsibilities
Design, develop, and optimize state-of-the-art NLP models for tasks such as named entity recognition (NER), text summarization, sentiment analysis, machine translation, and question answering.
Lead research and implementation of transformer-based architectures such as BERT, GPT, T5, LLaMA, and other large language models (LLMs).
Architect scalable NLP pipelines for real-time and batch processing of large volumes of unstructured text data.
Optimize NLP models for performance, including quantization, pruning, knowledge distillation, and efficient inference techniques.
Deploy NLP models in production using cloud-native solutions such as AWS SageMaker, GCP Vertex AI, Azure ML, or on-premise infrastructure.
Implement and maintain robust CI/CD pipelines for automated model training, deployment, and monitoring.
Develop advanced information retrieval and retrieval-augmented generation (RAG) techniques using vector databases such as FAISS, Pinecone, or Weaviate.
Integrate NLP models with search engines and recommendation systems to enhance user experiences.
Ensure NLP systems adhere to ethical AI principles, including fairness, interpretability, and bias mitigation.
Mentor junior engineers and collaborate with cross-functional teams, including data scientists, MLOps engineers, and software developers.
Required Skills
Programming: Expert-level proficiency in Python, with strong experience in NLP libraries such as Hugging Face Transformers, SpaCy, NLTK, and FastText.
NLP Techniques: Deep understanding of tokenization, embeddings, attention mechanisms, sequence-to-sequence models, and transfer learning.
Model Fine-Tuning: Proven experience in fine-tuning and optimizing transformer-based NLP models (BERT, GPT, T5, LLaMA, etc.).
Model Deployment: Strong experience in deploying NLP models at scale using Flask, FastAPI, Kubernetes, or cloud-based ML services.
Scalability & Performance: Expertise in optimizing NLP models for low-latency, high-throughput environments, including hardware acceleration techniques (TPUs, GPUs).
Data Engineering: Experience handling massive text datasets, data pipelines (Spark, Dask, Apache Airflow), and vector databases.
MLOps & CI/CD: Strong knowledge of experiment tracking, model versioning, and automation using MLflow, Weights & Biases, or DVC.
Cloud & Infrastructure: Hands-on experience with cloud platforms (AWS, GCP, Azure) and serverless architectures for NLP workloads.
Security & Compliance: Knowledge of data privacy regulations (GDPR, CCPA) and responsible AI best practices for NLP applications.
Nice to Have
Experience with multimodal AI, integrating NLP with vision and speech models.
Familiarity with graph-based NLP techniques, knowledge graphs, and ontology-based text processing.
Experience in building conversational AI and chatbot systems using Rasa or OpenAI API.
Knowledge of A/B testing and user feedback loops for NLP model improvements.
Strong publication record or contributions to open-source NLP projects.
"""
}

cv_engineer = {
    "jr": """The selected candidate will be responsible for developing, fine-tuning, and deploying computer vision models. The role involves working with image and video data, implementing state-of-the-art deep learning techniques, and integrating models into production systems.
Responsibilities
Assist in preprocessing and augmenting image and video data, including resizing, normalization, and data augmentation techniques.
Develop and fine-tune computer vision models using libraries such as OpenCV, TensorFlow, PyTorch, or MMDetection.
Implement feature extraction techniques such as HOG, SIFT, ORB, and CNN-based embeddings.
Train and evaluate models for tasks like image classification, object detection, image segmentation, and face recognition.
Deploy computer vision models as APIs using Flask or FastAPI.
Optimize vision pipelines for efficiency and scalability in production environments.
Work with large-scale image/video datasets, performing exploratory data analysis (EDA) to extract insights.
Collaborate with data engineers to integrate computer vision solutions into existing data pipelines.
Implement model evaluation metrics such as IoU, mAP, precision, and recall for assessing model performance.
Maintain clear documentation for models, APIs, and data preprocessing techniques.
Required Skills
Programming: Proficiency in Python with experience in libraries like NumPy, Pandas, and OpenCV.
Computer Vision Fundamentals: Understanding of image processing techniques, feature extraction, and deep learning architectures for vision tasks.
Model Development: Hands-on experience with deep learning frameworks such as TensorFlow, PyTorch, or Keras for computer vision applications.
Image Preprocessing: Ability to clean and prepare image and video data for training and inference.
Model Deployment: Exposure to deploying computer vision models as REST APIs using Flask or FastAPI.
Version Control & CI/CD: Familiarity with Git and basic CI/CD concepts.
Cloud Basics: Some exposure to vision-related services in AWS, GCP, or Azure.
Problem-Solving: Strong analytical skills with the ability to troubleshoot computer vision pipeline issues.
Nice to Have
Experience with transformer-based vision models such as Vision Transformers (ViTs) or CLIP.
Knowledge of multi-object tracking, pose estimation, and video analytics.
Familiarity with large-scale image processing using Apache Spark or Dask.
Understanding of deploying models using Docker and Kubernetes.
Exposure to edge computing for real-time vision applications.
""",
    "mid": """Computer Vision Engineer (3-5 Years Experience) – Job Description
The selected candidate will be responsible for designing, developing, and deploying advanced computer vision models for real-world applications. The role involves working with image and video data, leveraging deep learning and traditional computer vision techniques, optimizing model performance, and integrating solutions into production environments.
Responsibilities
Develop and optimize computer vision models for tasks such as image classification, object detection, instance/semantic segmentation, and pose estimation.
Work with deep learning frameworks such as TensorFlow, PyTorch, or ONNX for training and deploying models.
Implement traditional image processing techniques using OpenCV, Scikit-Image, or custom feature engineering methods.
Build and fine-tune models using pre-trained architectures like ResNet, EfficientNet, YOLO, Faster R-CNN, Mask R-CNN, and Vision Transformers (ViTs).
Process and augment large-scale image and video datasets to improve model robustness.
Deploy computer vision models as REST APIs using Flask, FastAPI, or gRPC.
Optimize models for efficiency using techniques such as model quantization, pruning, and TensorRT acceleration.
Integrate computer vision pipelines with cloud-based services such as AWS Rekognition, GCP Vision AI, or Azure Computer Vision.
Work with edge computing frameworks for real-time vision applications on embedded devices (e.g., NVIDIA Jetson, OpenVINO, TensorFlow Lite).
Implement monitoring and model drift detection techniques to ensure model reliability in production.
Collaborate with data engineers and software developers to integrate vision solutions into enterprise applications.
Maintain well-documented code, APIs, and research findings.
Required Skills
Programming: Strong proficiency in Python and experience with OpenCV, NumPy, Pandas, and Matplotlib.
Deep Learning & Model Development: Hands-on experience with CNN architectures, transfer learning, and transformer-based vision models (e.g., ViTs, DINO).
Computer Vision Fundamentals: Expertise in feature extraction, keypoint detection, image segmentation, and 3D vision techniques.
Model Deployment: Experience in deploying models using Flask, FastAPI, Docker, or Kubernetes.
Optimization & Acceleration: Understanding of model quantization, TensorRT, OpenVINO, and hardware acceleration for real-time performance.
Cloud & Edge Computing: Familiarity with AWS, GCP, or Azure vision services, and edge computing frameworks.
MLOps & CI/CD: Experience with version control (Git), CI/CD pipelines, and model lifecycle management tools (e.g., MLflow, DVC).
Data Processing & Augmentation: Strong knowledge of image augmentation techniques (Albumentations, PIL) and dataset preparation.
Problem-Solving: Strong analytical and troubleshooting skills to optimize model performance and deployment.
Nice to Have
Experience with multi-object tracking (MOT), pose estimation, and 3D reconstruction.
Knowledge of self-supervised learning and foundation models for vision tasks.
Exposure to reinforcement learning for vision-based decision-making.
Familiarity with Apache Spark or Dask for distributed image processing.
Experience in developing vision applications for AR/VR or robotics.
""",
    "sr": """Computer Vision Engineer (5+ Years Experience) – Job Description
The selected candidate will be responsible for architecting, developing, and deploying cutting-edge computer vision solutions for large-scale real-world applications. This role involves working with image and video data, designing deep learning architectures, optimizing model performance, and leading the integration of vision-based AI systems into production environments. The candidate is expected to provide technical leadership, mentor junior engineers, and drive innovation in computer vision projects.
Responsibilities
Design and implement state-of-the-art computer vision models for tasks such as object detection, image segmentation, action recognition, 3D reconstruction, and multi-object tracking (MOT).
Lead research and development of novel computer vision techniques using deep learning frameworks like TensorFlow, PyTorch, or JAX.
Develop scalable and efficient computer vision pipelines, optimizing models for real-time performance using techniques such as quantization, pruning, distillation, and TensorRT acceleration.
Work with large-scale image and video datasets, performing preprocessing, augmentation, and feature extraction to enhance model robustness.
Deploy production-ready computer vision models using microservices architecture, containerization (Docker, Kubernetes), and cloud platforms (AWS, GCP, Azure).
Architect and optimize ML pipelines for edge computing applications (e.g., NVIDIA Jetson, OpenVINO, TensorFlow Lite) for low-latency inference.
Implement self-supervised and foundation models (Vision Transformers, CLIP, DINO) to enhance feature learning for vision tasks.
Collaborate with cross-functional teams, including data engineers, software developers, and product managers, to integrate vision AI models into enterprise applications.
Lead MLOps practices by setting up CI/CD pipelines, model versioning, monitoring, and logging using MLflow, DVC, or similar tools.
Develop scalable and distributed training solutions using multi-GPU frameworks (Horovod, DeepSpeed) and cloud computing.
Stay up to date with the latest advancements in computer vision, deep learning, and AI research, and drive their adoption in projects.
Provide technical mentorship and guidance to junior engineers and data scientists.
Document research findings, model architectures, and deployment strategies for knowledge sharing and reproducibility.
Required Skills
Programming: Expert proficiency in Python and strong experience with OpenCV, NumPy, Pandas, and SciPy.
Deep Learning & Model Development: Hands-on experience with advanced architectures such as Vision Transformers (ViTs), Swin Transformers, EfficientNet, YOLO, and Mask R-CNN.
Computer Vision Fundamentals: Strong expertise in feature extraction, keypoint detection, multi-view geometry, optical flow, and image enhancement.
Model Deployment & Optimization: Experience with deploying computer vision models in production using Flask, FastAPI, gRPC, TensorRT, and OpenVINO.
Scalability & Performance: Strong understanding of distributed training, model parallelism, and optimization techniques for large-scale applications.
Cloud & Edge AI: Deep knowledge of deploying models on AWS, GCP, or Azure, and experience with edge computing solutions.
MLOps & Automation: Strong experience with MLflow, DVC, CI/CD, Kubernetes, and automated model monitoring.
3D Vision & AR/VR: Familiarity with depth estimation, SLAM, LiDAR processing, and 3D reconstruction.
Leadership & Collaboration: Proven ability to lead projects, mentor engineers, and collaborate with cross-functional teams.
Nice to Have
Experience with multi-camera vision systems, LiDAR, and sensor fusion techniques.
Knowledge of reinforcement learning for vision-based decision-making.
Exposure to generative AI models for vision tasks (e.g., Stable Diffusion, GANs).
Experience with Apache Spark or Dask for large-scale distributed image processing.
Research publications or patents in computer vision and deep learning.
"""
}