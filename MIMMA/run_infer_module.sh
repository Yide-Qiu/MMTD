
# python infer_test.py --dataset_name SatSOT --exp_name SatSOT_MFTP --module MFTP --thresh_prob_ablation 0 --ckpt_dir SatSOT_MFTP --thresh_prob 0.2 --thresh_ground 0.2
# python infer_test.py --dataset_name VISO --exp_name VISO_MFTP --module MFTP --thresh_prob_ablation 0 --ckpt_dir VISO_MFTP --thresh_prob 0.5 --thresh_ground 0.2
# python infer_test.py --dataset_name OOTB --exp_name OOTB_MFTP --module MFTP --thresh_prob_ablation 0 --ckpt_dir OOTB_MFTP --thresh_prob 0.2 --thresh_ground 0.2
# python infer_test.py --dataset_name OTB100 --exp_name OTB100_MFTP --module MFTP --thresh_prob_ablation 0 --ckpt_dir OTB100_MFTP --thresh_prob 0.4 --thresh_ground 0.03
# python infer_test.py --dataset_name MSTAv2 --exp_name MSTAv2_MFTP --module MFTP --thresh_prob_ablation 0 --ckpt_dir MSTAv2_MFTP

# python infer_test.py --dataset_name SatSOT --exp_name SatSOT_MSTA --module MSTA --thresh_prob_ablation 0 --ckpt_dir SatSOT_MSTA --thresh_prob 0.2 --thresh_ground 0.02
# python infer_test.py --dataset_name VISO --exp_name VISO_MSTA --module MSTA --thresh_prob_ablation 0 --ckpt_dir VISO_MSTA --thresh_prob 0.5 --thresh_ground 0.05
# python infer_test.py --dataset_name OOTB --exp_name OOTB_MSTA --module MSTA --thresh_prob_ablation 0 --ckpt_dir OOTB_MSTA --thresh_prob 0.2 --thresh_ground 0.02
# python infer_test.py --dataset_name OTB100 --exp_name OTB100_MSTA --module MSTA --thresh_prob_ablation 0 --ckpt_dir OTB100_MSTA --thresh_prob 0.4 --thresh_ground 0.03

# python infer_test.py --dataset_name SatSOT --exp_name SatSOT_MIMMA --module MIMMA --thresh_prob_ablation 0 --ckpt_dir SatSOT_MIMMA --thresh_prob 0.2 --thresh_ground 0.02
# python infer_test.py --dataset_name VISO --exp_name VISO_MIMMA --module MIMMA --thresh_prob_ablation 0 --ckpt_dir VISO_MIMMA --thresh_prob 0.5 --thresh_ground 0.05
# python infer_test.py --dataset_name OOTB --exp_name OOTB_MIMMA --module MIMMA --thresh_prob_ablation 0 --ckpt_dir OOTB_MIMMA --thresh_prob 0.2 --thresh_ground 0.02
# python infer_test.py --dataset_name OTB100 --exp_name OTB100_MIMMA --module MIMMA --thresh_prob_ablation 0 --ckpt_dir OTB100_MIMMA --thresh_prob 0.4 --thresh_ground 0.03

# cd ..
# cp -r MSTA/outputs/VISO/VISO_MFTP/ sota/UCMCTrack-master/output/
# cp -r MSTA/outputs/SatSOT/SatSOT_MFTP/ sota/UCMCTrack-master/output/
# cp -r MSTA/outputs/OOTB/OOTB_MFTP/ sota/UCMCTrack-master/output/
# cp -r MSTA/outputs/OTB100/OTB100_MFTP/ sota/UCMCTrack-master/output/

# cp -r MSTA/outputs/VISO/VISO_MSTA/ sota/UCMCTrack-master/output/
# cp -r MSTA/outputs/SatSOT/SatSOT_MSTA/ sota/UCMCTrack-master/output/
# cp -r MSTA/outputs/OOTB/OOTB_MSTA/ sota/UCMCTrack-master/output/
# cp -r MSTA/outputs/OTB100/OTB100_MSTA/ sota/UCMCTrack-master/output/

# cp -r MSTA/outputs/VISO/VISO_MIMMA/ sota/UCMCTrack-master/output/
# cp -r MSTA/outputs/SatSOT/SatSOT_MIMMA/ sota/UCMCTrack-master/output/
# cp -r MSTA/outputs/OOTB/OOTB_MIMMA/ sota/UCMCTrack-master/output/
# cp -r MSTA/outputs/OTB100/OTB100_MIMMA/ sota/UCMCTrack-master/output/



