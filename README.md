# Clip-DRR
AsyDyReC: A Multimodal Graph-based Framework for Spatio-Temporal Asymmetric Dyadic Relationship Classification

## Environment
Please refer to the "environment.txt".

## Dataset
[NoXi](https://multimediate-challenge.org/datasets/Dataset_NoXi/) was released by [[Angelo et al. ICMI 2017](https://dl.acm.org/doi/abs/10.1145/3136755.3136780?casa_token=8UoDP_iZs3gAAAAA:mYUOHpnezNatC2FQpxXXur2Y8CWiKmS_2Jech1yxEp-XcBU9OsrgC6li0zdN5Up9ornfGimLOv4)]. It involves 4 different relationships: Stranger, Acquaintance, Friend and Very good friend.

[UDIVA](https://chalearnlap.cvc.uab.cat/dataset/41/description/) was released by [[Cristina et al. WACV 2021](https://openaccess.thecvf.com/content/WACV2021W/HBU/html/Palmero_Context-Aware_Personality_Inference_in_Dyadic_Scenarios_Introducing_the_UDIVA_Dataset_WACVW_2021_paper.html)]. It involves 2 relationships: Unknown and Known.

## Usage
    optional arguments:
      --mode         Choose NoXi or UDIVA
      --person       For the asymmetric relationship in NoXi, choose either A or B
      --k_fold       K-fold cross validation
      --workers N         number of data loading workers (defult: 4)
      --batch-size N      mini-batch size (default: 1)
      --print-freq N, -p N      print frequency (default: 10)
      -n N, --num-classes N     number of classes
      --total_frames       The number of frames contained in each clip


## Contributing
For any questions, feel free to open an issue or contact us (tangwang@stu.scu.edu.cn)
