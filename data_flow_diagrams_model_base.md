Listed directory project
Viewed generator.py:1-249
Viewed discriminator.py:1-147
Viewed training_loop.py:1-410

Here are the data flow and architecture diagrams for your CycleGAN's `generator.py`, `discriminator.py`, and `training_loop.py`. These visualize the tensor shapes at each critical transformation stage based on your PyTorch implementations.

### 1. Generator (`generator.py`)
This diagram shows the ResNet-based generator architecture which handles translation between domains while preserving structural features.

```mermaid
graph TD
    In["Input Tensor<br/>Shape: (B, 3, 256, 256)"]

    subgraph "Encoder"
        C1["Reflection Pad + 7x7 Conv + IN + ReLU<br/>Shape: (B, 64, 256, 256)"]
        D1["3x3 Conv (stride 2) + IN + ReLU<br/>Shape: (B, 128, 128, 128)"]
        D2["3x3 Conv (stride 2) + IN + ReLU<br/>Shape: (B, 256, 64, 64)"]
    end

    subgraph "Transformer Components"
        R1["9x ResNet Blocks<br/>(Skip connections preserve structure)<br/>Shape: (B, 256, 64, 64)"]
    end

    subgraph "Decoder"
        U1["3x3 ConvTranspose (stride 2) + IN + ReLU<br/>Shape: (B, 128, 128, 128)"]
        U2["3x3 ConvTranspose (stride 2) + IN + ReLU<br/>Shape: (B, 64, 256, 256)"]
        OutLayer["Reflection Pad + 7x7 Conv + Tanh<br/>Shape: (B, 3, 256, 256)"]
    end

    In --> C1
    C1 --> D1
    D1 --> D2
    D2 --> R1
    R1 --> U1
    U1 --> U2
    U2 --> OutLayer
```

### 2. Discriminator (`discriminator.py`)
This is the PatchGAN architecture, evaluating whether overlapping patches of the image are real or generated.

```mermaid
graph TD
    In["Input Image (Real or Fake)<br/>Shape: (B, 3, 256, 256)"]

    subgraph "Downsampling Path"
        C1["4x4 Conv (stride 2) + LeakyReLU<br/>Shape: (B, 64, 128, 128)"]
        C2["4x4 Conv (stride 2) + IN + LeakyReLU<br/>Shape: (B, 128, 64, 64)"]
        C3["4x4 Conv (stride 2) + IN + LeakyReLU<br/>Shape: (B, 256, 32, 32)"]
    end

    subgraph "Output Formatting"
        C4["4x4 Conv (stride 1) + IN + LeakyReLU<br/>Shape: (B, 512, 31, 31)"]
        Out["Final 4x4 Conv (stride 1)<br/>Shape: (B, 1, 30, 30)"]
    end

    In --> C1
    C1 --> C2
    C2 --> C3
    C3 --> C4
    C4 --> Out
```
> [!NOTE]
> The final output `(B, 1, 30, 30)` interprets a single probability score for each 70x70 receptive field location across the `256x256` image.

### 3. Training Loop (`training_loop.py`)
This details how data flows back and forth to compute differences in CycleGAN's composite losses, specifically noting adversarial and cycle combinations.

```mermaid
graph TD
    subgraph "Input Data"
        RealA["Real Image Domain A<br/>(B, 3, 256, 256)"]
        RealB["Real Image Domain B<br/>(B, 3, 256, 256)"]
    end

    subgraph "Stage 1: Generator Forward Pass"
        RealA -->|Input| G_AB["Generator A->B"]
        RealB -->|Input| G_BA["Generator B->A"]
        
        G_AB --> FakeB["Fake Image B<br/>(B, 3, 256, 256)"]
        G_BA --> FakeA["Fake Image A<br/>(B, 3, 256, 256)"]
        
        FakeA -->|Input| G_AB_Cycle["Generator A->B (Cycle)"]
        FakeB -->|Input| G_BA_Cycle["Generator B->A (Cycle)"]
        
        G_AB_Cycle --> RecB["Reconstructed B<br/>(B, 3, 256, 256)"]
        G_BA_Cycle --> RecA["Reconstructed A<br/>(B, 3, 256, 256)"]
    end

    subgraph "Stage 2: Generator Updates"
        direction LR
        FakeB --> D_B_eval["Discriminator B"]
        FakeA --> D_A_eval["Discriminator A"]
        
        D_B_eval -->|fake score| G_Loss["Generator Composite Loss<br/>1. Adversarial Loss<br/>2. Cycle Loss (Real vs Rec)<br/>3. Identity & Perceptual losses"]
        D_A_eval -->|fake score| G_Loss
        RecA --> G_Loss
        RecB --> G_Loss
        RealA -.-> G_Loss
        RealB -.-> G_Loss
        
        G_Loss --> G_Opt["Update G_AB & G_BA via GradScaler"]
    end

    subgraph "Stage 3: Discriminator Updates"
        direction TB
        BufferB["Pool/Buffer of Fake B"]
        BufferA["Pool/Buffer of Fake A"]
        
        FakeB -.->|Store| BufferB
        FakeA -.->|Store| BufferA
        
        RealA --> D_A_train["Discriminator A"]
        BufferA --> D_A_train
        D_A_train --> D_A_Loss["Loss_D_A (Real vs Fake)<br/>Shape at evaluation: (B, 1, 30, 30)"]
        D_A_Loss --> D_A_Opt["Update D_A via GradScaler"]

        RealB --> D_B_train["Discriminator B"]
        BufferB --> D_B_train
        D_B_train --> D_B_Loss["Loss_D_B (Real vs Fake)<br/>Shape at evaluation: (B, 1, 30, 30)"]
        D_B_Loss --> D_B_Opt["Update D_B via GradScaler"]
    end
    
    %% Connections ordering flows naturally from steps
    Stage1 --> Stage2
    Stage2 --> Stage3
    
    style G_Loss fill:#c44d,stroke:#333
    style D_A_Loss fill:#3c4d,stroke:#333
    style D_B_Loss fill:#3c4d,stroke:#333
```