# MMARN

A multimodal medical image registration method based on modal transformation.



###### The environmental dependencies are listed in the requirements.txt file. The training entry point and test entry point are main.py and test.py respectively. The pre-trained parameters are stored in the weight folder.

###### It should be noted that although MMARN has multiple loss functions, during training, it is only necessary to ensure that these loss functions have consistent dimensions at the initial stage. Due to the inherent flaws of GAN, the discriminator may sometimes be too strong or too weak to guide the generative model. In such cases, it is only necessary to adjust the optimization interval of the discriminator, i.e., to ensure that the loss of the discriminator and the loss of the generator are in the same dimension.





# Citation Information



Title: Unsupervised Abdominal Multimodal Image Registration based on Modality and Morphological Alignment

(https://www.sciencedirect.com/science/article/abs/pii/S0957417426003027)



Abstract: Renal tumors represent malignant neoplasms within the urogenital system, exhibiting distinct imaging features across different modalities. Aligning the spatial positions of tumors across different modalities and integrating multi-modal image features holds promise for enhancing the accuracy of downstream tasks such as clinical diagnostic analysis. However, the attention paid to abdominal tumors in current registration studies remains relatively limited. Existing cross-modal registration methods based on modality transformation lack effective control over the modality transformation module, which limits the accuracy of registration tasks relying on transformed images. To address this, this paper proposes a multimodal registration method based on morphology-consistent and modality-consistent transformation: it decouples the modal and morphological features of images using prototype loss to improve the representational capability of modalities, and employs multi-scale consistency-based modal consistency loss and morphological consistency loss to ensure that modality-transformed images preserve the anatomical structures of the original images. Moreover, due to the relatively small size of tumors, their impact on the loss function is minimal, making them prone to being overlooked by the model. Meanwhile, respiratory motion in the abdomen leads to larger deformation amplitudes, which are difficult to handle with a single deformation field. Therefore, this paper proposes a cascaded registration module based on error accumulation. It adaptively copes with large-scale deformations by cascading multiple sub-registration networks and enhances the model’s focus on hard-to-register regions such as tumors through error accumulation. We compared the proposed method with recently introduced registration methods in kidney tumor registration and brain tumor registration. The experimental results show that our method outperforms the competing methods in all cases. Additionally, we verified that our method has superior modality transformation capability.



Keywords: Abdomen tumor registration; Unsupervised Learning; Morphological Alignment; Modality Alignment; Cascade registration



Cite this article: Kanqi Wang, Ziyang Mei, Siyuan Han, Lianting Zhong, Yuqing Ding, Jianjun Zhou, Yang Zhao, Gang Liu. Unsupervised Abdominal Multimodal Image Registration based on Modality and Morphological Alignment. Expert Systems with Applications. 2026, 131389

