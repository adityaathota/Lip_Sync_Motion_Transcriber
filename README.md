# LiSMoT: Lip_Sync_Motion_Transcriber

![lipreading](https://github.com/user-attachments/assets/02802104-5ebf-4453-aada-7914e8d4d589)

## Overview and Purpose

* LiSMoT is an end-to-end deep learning system designed to perform automatic lip reading from video sequences
* The project aims to decode spoken language by visually interpreting lip movements without requiring audio input
* The system can be used to enhance accessibility for people with hearing impairments and improve communication in noisy environments
* It represents an application of computer vision and sequence modeling in practical human-computer interaction scenarios

## Data Acquisition and Management

* The system uses specialized lip reading datasets containing video clips of individuals speaking with corresponding text annotations
* Data is downloaded automatically using gdown from Google Drive in the format of video files (.mpg) and alignment files (.align)
* The dataset is structured into video files of speakers and corresponding text alignment files
* The data is split into training and testing sets with approximately 450 samples for training and 50 for testing

## Preprocessing Pipeline

* Video frames undergo multiple preprocessing steps:
* Conversion to grayscale to reduce dimensionality and computational complexity
* Cropping to isolate and focus on the mouth region (frames[190:236,80:220,:])
* Normalization by subtracting the mean and dividing by standard deviation to handle lighting variations
* Statistical analysis to ensure data quality and consistency
* Text processing uses a defined vocabulary of characters, numbers, and special symbols
* Character-to-number mapping is created for tokenization of text data
* Special padding is applied to handle variable-length sequences in batches

## Model Architecture

* The neural network consists of three main components:

### 1. Spatiotemporal Feature Extraction (3D CNN Layers)
* Three stacked 3D Convolutional Neural Network layers:
* First layer: 128 filters with 3×3×3 kernel size, input shape: (75, 46, 140, 1)
* Second layer: 256 filters with 3×3×3 kernel size
* Third layer: 75 filters with 3×3×3 kernel size
* Each layer uses ReLU activation and 3D Max Pooling with pool size (1, 2, 2)
* These layers capture both spatial and temporal patterns in lip movements simultaneously

### 2. Sequence Modeling (Bidirectional LSTM Layers)
* Time-Distributed Flatten layer to preserve temporal structure
* Two Bidirectional LSTM layers with 128 units in each direction
* Orthogonal kernel initialization for improved convergence
* 50% dropout between layers for regularization
* Processes sequences in both forward and backward directions to capture context

### 3. Classification Layer
* Dense output layer with softmax activation
* Output units: vocabulary size + 1 for CTC blank token (41 units total)
* He normal kernel initialization

## Training Methodology

* Connectionist Temporal Classification (CTC) loss function:
* Addresses the alignment problem between input frames and output characters
* Handles variable-length sequences effectively
* Uses blank tokens to model transitions between characters
* Eliminates the need for explicit alignment between frames and text

* Optimization strategy:
* Adam optimizer with an initial learning rate of 0.0001
* Learning rate scheduling: constant for the first 30 epochs, then exponential decay
* Batch size of 2 for training to fit memory constraints
* Multiple callbacks for checkpointing, learning rate scheduling, and monitoring

* Model parameters:
* Total trainable parameters: 8,471,924
* 3D CNN layers: 1,407,051 parameters
* Bidirectional LSTM layers: 7,054,336 parameters (83% of total)
* Dense output layer: 10,537 parameters

## Inference Process

* Video input is processed through the same preprocessing pipeline as training
* Processed frames are passed through the 3D CNN layers for feature extraction
* Extracted features are processed by Bidirectional LSTM layers for sequence modeling
* The model outputs character probabilities at each time step
* CTC decoding (either greedy or beam search) converts probabilities into character sequences
* Numerical indices are mapped back to characters to form the final text output

## Theoretical Innovations

* 3D CNNs vs 2D CNNs: Using 3D convolutions allows simultaneous modeling of spatial and temporal dimensions
* Bidirectional processing: Captures both past and future contextual information
* End-to-end training: The entire pipeline from video processing to text prediction is trained jointly
* Mathematical formulation of 3D CNN operation:
  * [3DCNN(𝑥, 𝑤)]𝑐0𝑡𝑖𝑗= ∑𝑐=1𝐶∑𝑡0=1𝑘𝑡∑𝑖0=1𝑘𝑤∑𝑗0=1𝑘ℎ𝑤𝑐0𝑐𝑡0𝑖0𝑗0𝑥𝑐,𝑡+𝑡0,𝑖+𝑖0,𝑗+𝑗0

## Performance and Results

* Implementation in TensorFlow ensures efficient computation and model training
* Performance metrics vary across implementations, with accuracy rates between:
  * 85.65% transcription accuracy reported in some implementations
  * Word Error Rate (WER) of approximately 20.33%
  * Character Error Rate (CER) of approximately 15.17%
* The system demonstrates robustness in handling speaker variations and lighting conditions

## Challenges in Lip Reading

* Homophenes: Visually similar phonemes that look identical on the lips (e.g., 'p', 'b', and 'm')
* Coarticulation effects: Lip movements for one sound affected by preceding and following sounds
* Speaker variability: Different speaking styles, accents, and facial structures
* Temporal dynamics: Variations in speaking speed and rhythm
* Environmental factors: Lighting conditions, camera angles, and video quality

## Applications and Use Cases

* Accessibility services for deaf and hard-of-hearing individuals
* Security and surveillance in audio-restricted environments
* Communication in noisy settings where audio is unreliable
* Human-computer interaction enhancement
* Crime investigation and forensic analysis of video evidence
* Educational tools for language learning and speech therapy

## Future Improvements

* Multi-modal integration: Combining visual features with audio when available
* Attention mechanisms: Focusing on the most informative regions and time steps
* Transformer architectures: Replacing recurrent components with self-attention
* Few-shot learning: Adapting to new speakers with minimal examples
* Real-time processing: Optimizing for low-latency applications
* Expanding vocabulary and language support

## System Requirements and Implementation

* Dependencies: OpenCV, TensorFlow, Matplotlib, imageio, gdown, numpy
* Hardware requirements: GPU recommended for efficient training
* Input: Video files focusing on speaker's face
* Output: Text transcription of spoken words
* Model deployment options include saved checkpoints for inference on new videos

Citations:
1. [LiSMoT.ipynb](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/53782366/7bdb0404-9f3b-4471-8383-74d80b2afa91/LiSMoT.ipynb)
2. [IJISRT - Visual Speech Recognition](https://ijisrt.com/assets/upload/files/IJISRT23AUG1551.pdf)
3. [imanager - Paper on Deep Learning](https://www.imanagerpublications.com/article/20576/)
4. [CEUR-WS - Paper on Deep Learning](https://ceur-ws.org/Vol-3868/Paper8.pdf)
5. [IJARET - Paper on Speech Recognition](https://iaeme.com/MasterAdmin/Journal_uploads/IJARET/VOLUME_11_ISSUE_12/IJARET_11_12_198.pdf)
6. [TechScience - CMC Paper on Deep Learning](https://www.techscience.com/cmc/v68n2/42177/html)
7. [Visual Speech Recognition Using 3D CNN and LSTMs](https://restpublisher.com/wp-content/uploads/2024/09/Visual-Speech-Recognition-Using-3D-CNN-and-LSTMS.pdf)
8. [IJCRT Paper on Speech Recognition](https://ijcrt.org/papers/IJCRT2406297.pdf)
9. [KSCST - BE Project on Speech](https://www.kscst.org.in/spp/47_series/47s_spp/Exhibition%20Projects/201_47S_BE_3986.pdf)
10. [Wiley - Chapter on Speech Recognition](https://onlinelibrary.wiley.com/doi/abs/10.1002/9781119792826.ch4)
11. [IJETMS Paper on Speech Processing](https://ijetms.in/Vol-8-issue-3/Vol-8-Issue-3-39.pdf)
12. [CSEIJ Paper on Deep Learning for Speech](https://www.cseij.org/papers/v15n1/15125cseij10.pdf)
13. [MDPI Paper on Deep Learning for Vision and Speech](https://www.mdpi.com/2076-3417/12/19/9532)
14. [Lipreading Task on Papers with Code](https://paperswithcode.com/task/lipreading)
15. [MDPI - Paper on Speech and Video Processing](https://www.mdpi.com/1424-8220/22/1/72)
16. [Scope Journal - Deep Learning Paper](https://scope-journal.com/assets/uploads/doc/a7736-326-336.202410116.pdf)
17. [Semantic Scholar Paper on Speech Recognition](https://www.semanticscholar.org/paper/b383e01326c587a97e776d939f127a180e54b563)
18. [GitHub Repository for Lip Reading Deep Learning](https://github.com/astorfi/lip-reading-deeplearning)
19. [MDPI - Paper on Visual Speech Recognition](https://www.mdpi.com/2673-7426/4/1/23)
20. [Blog on Multi-Modal Methods](https://blog.mlreview.com/multi-modal-methods-part-one-49361832bc7e)
21. [ArXiv Paper on Lipreading](https://arxiv.org/abs/2110.07879)

