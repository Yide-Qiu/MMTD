
python infer_test.py --dataset_name SatSOT --exp_name SatSOT_MIMMA --module MIMMA --thresh_ground_ablation 1 --ckpt_dir SatSOT_MIMMA
python infer_test.py --dataset_name VISO --exp_name VISO_MIMMA --module MIMMA --thresh_ground_ablation 1 --ckpt_dir VISO_MIMMA
python infer_test.py --dataset_name OOTB --exp_name OOTB_MIMMA --module MIMMA --thresh_ground_ablation 1 --ckpt_dir OOTB_MIMMA
python infer_test.py --dataset_name OTB100 --exp_name OTB100_MIMMA --module MIMMA --thresh_ground_ablation 1 --ckpt_dir OTB100_MIMMA
# python infer_test.py --dataset_name MSTAv2 --exp_name MSTAv2_MIMMA --module MIMMA --thresh_ground_ablation 1 --ckpt_dir MSTAv2_MIMMA

# all test ok and eval ok

# cd ..
# # cp -r MSTA/outputs/SatSOT/SatSOT_PMST/ sota/UCMCTrack-master/output/
# # cp -r MSTA/outputs/AccessAIS/AccessAIS_PMST/ sota/UCMCTrack-master/output/
# # cp -r MSTA/outputs/MSTAv2/MSTAv2_MFTP/ sota/UCMCTrack-master/output/
# # cp -r MSTA/outputs/MSTAv2/MSTAv2_MSTA/ sota/UCMCTrack-master/output/
# cp -r MSTA/outputs/MSTAv2/MSTAv2_MIMMA/ sota/UCMCTrack-master/output/



