# Dovetail


[Dovetail](https://arxiv.org/pdf/2412.18934) combines speculative decoding characteristics with heterogeneous architecture, proposing a scheme to deploy the target model on the CPU and the smaller model on the GPU. This approach fully utilizes the hardware resources of existing consumer-grade devices, leveraging the respective strengths of CPUs and GPUs, thereby significantly improving the inference speed of the target model and providing strong support for its efficient operation on consumer-grade devices.


## Setup & Installation


```bash
git clone https://github.com/ddInference/Dovetail.git
cd Dovetail
pip install -r requirements.txt
```

## The default main branch is the implementation of Dovetail. If you want to run Dovetail on a laptop, please use its 8-bit quantization by switching to the **Dovetail_quantify** branch.



## Dovetail Weights

 w/o both indicates that neither DGF nor additional layers are used, w/ DGF indicates that only DGF is used, and w/ DGF + m indicates that DGF is used along with the addition of m Transformer blocks.

| Base Model         | HCSD Weights                                                 |
| ------------------ | ------------------------------------------------------------ |
| w/o both        | - | 
| w/ DGF          | - | 
| w/ DGF + 1      | - | 
| w/ DGF + 2      | - | 
| w/ DGF + 3      | - | 
| w/ DGF + 4      | - | 
| w/ DGF + 5      | - | 

## Citation
Please cite our paper if you find the repo helpful:
```bibtex
@article{zhang2024dovetail,
  title={Dovetail: A CPU/GPU Heterogeneous Speculative Decoding for LLM inference},
  author={Zhang, Libo and Zhang, Zhaoning and Xu, Baizhou and Mei, Songzhu and Li, Dongsheng},
  journal={arXiv preprint arXiv:2412.18934},
  year={2024}
}
```



## Acknowledgements

This project has been influenced by many excellent projects in the LLM community, such as [EAGLE](https://github.com/SafeAILab/EAGLE) and others.
