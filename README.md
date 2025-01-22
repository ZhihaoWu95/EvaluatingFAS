##### Table of contents

1. [Environment setup](#environment-setup)
  
2. [Data preparation](#dataset-preparation)
  
3. [How to run](#how-to-run)
  

# Code implement for "Evaluating the Security of Commercial Face Authentication Systems Towards Physical Adversarial Attacks"

</div>

> **Abstract**:Face authentication systems are vital for securing access to confidential facilities, underscoring the paramount importance of their security. To counter the rising threat of physical adversarial attacks, real-world implementations of these systems often incorporate stringent security mechanisms and function as fully black-boxes, making existing attacks hardly effective and posing no immediate real-world threat. However, in this paper, we maintain a cautious perspective on the security of commercial systems. We conduct a comprehensive analysis of the security mechanisms employed by commercial systems and examine the attributes that contribute to their ability to counter existing attacks. Through it, we identify two key insights that enable bypassing those security mechanisms under the fully black-box setting and design a physical adversarial attack capable of spoofing commercial systems in the real world. Evaluations on three commercial face authentication SDKs (Face++, ArcSoft, Baidu) and five commercial access control devices (HikVision, DUMU, Dahua, Honeywell, and ZKTeco) reveal the realistic threat posed by our attacks, with an average success rate of 59%. Note that these products hold a global market share of nearly 50% (i.e., 3.3 Billion USD), and are widely used in various security-sensitive areas such as schools, hospitals, and banks, further emphasizing the real threat. We have reported the security exposure to relevant vendors. We also provide guidance for the development of robust security assessments and effective defense mechanisms to counter potential adversarial attacks.

## Environment setup

Create the environment:

```shell
conda create -n evalfas python=3.9 

conda activate evalfas 
```
then install dependencies

## Models

We provide one of the surrogate model links for testing. The full list will be released after acceptance.
You can download and implement the pretrained checkpoints of from MagFace[https://github.com/IrvingMeng/MagFace].


## Dataset

We have experimented with volunteers, but you can also use LFW for testing.

## How to run

To generate adversarial patches, you can run

```bash
cd attack
python demo_multimodel_glass.py
```
