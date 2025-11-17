## Environment Setup

```bash
conda create -n segdino python=3.10.16

conda activate segdino

pip install -r requirements.txt
````

Download [DINOv3](https://github.com/facebookresearch/dinov3)  pretrained weights

## Dataset Preparation


- **[TN3K](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation)**  
  A large-scale thyroid nodule segmentation dataset~\citep{gong2023thyroid}, containing **3,493 ultrasound images** with pixel-level annotations collected from multiple hospitals.  

- **[Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)**  
  A polyp segmentation dataset~\citep{jha2019kvasir} derived from colonoscopy examinations, consisting of **1,000 images** with high-quality expert annotations.  

- **[ISIC](https://challenge.isic-archive.com/data/#2017)**  
  A skin lesion segmentation benchmark~\citep{codella2018skin}, providing **2,750 dermoscopic images** annotated for lesion boundaries and covering diverse lesion types and acquisition conditions.  

Organize datasets in the following structure:

```
./segdata/tn3k
./segdata/kvasir
./segdata/isic
```

**NOTE: I have already preprocessed kvasir, so it can be used directly**

Each dataset folder should contain an  `images` directory and a  `masks` directory. However, you can change the base dir from the args.

## finetuning for kvasir


```bash
python lora_segdino.py --dino_ckpt /home/ahmedjaheen/SegDino/segdino/dinov3_vits16_pretrain_lvd1689m-08c60483.pth --dino_size s --dataset kvasir
```

