# python infer_test.py --dataset_name SatSOT --exp_name SatSOT_MIMMA
# python infer_test.py --dataset_name SAT_MTB --exp_name SAT_MTB_MIMMA --module MIMMA --thresh_prob_ablation 0 --ckpt_dir SAT_MTB_MIMMA --thresh_prob 0.5 --thresh_ground 0.04
python infer_test.py --dataset_name MTAD --exp_name MTAD_MIMMA --module MIMMA --thresh_prob_ablation 0 --ckpt_dir MTAD_MIMMA --thresh_prob 0.5 --thresh_ground 0.04
# python infer_test.py --dataset_name OOTB --exp_name OOTB_MIMMA
# python infer_test.py --dataset_name OTB100 --exp_name OTB100_MIMMA
# python infer_test.py --dataset_name VISO --exp_name VISO_MIMMA
python infer_test.py --dataset_name AccessAIS --exp_name AccessAIS_MIMMA --module MIMMA --thresh_prob_ablation 0 --ckpt_dir AccessAIS_MIMMA --thresh_prob 0.5 --thresh_ground 0.04
# python infer_test.py --dataset_name MSTAv2 --exp_name MSTAv2_MIMMA --module MIMMA --thresh_prob_ablation 0 --ckpt_dir MSTAv2_MIMMA --thresh_prob 0.5 --thresh_ground 0.04

cd ..
# cp -r MSTA/outputs/OOTB/OOTB_MIMMA/ sota/UCMCTrack-master/output/
# cp -r MSTA/outputs/OTB100/OTB100_MIMMA/ sota/UCMCTrack-master/output/
# cp -r MSTA/outputs/MSTAv2/MSTAv2_MIMMA/ sota/UCMCTrack-master/output/
cp -r MSTA/outputs/AccessAIS/AccessAIS_MIMMA/ sota/UCMCTrack-master/output/
# cp -r MSTA/outputs/SAT_MTB/SAT_MTB_MIMMA/ sota/UCMCTrack-master/output/
cp -r MSTA/outputs/MTAD/MTAD_MIMMA/ sota/UCMCTrack-master/output/
# cp -r MSTA/outputs/SatSOT/SatSOT_MIMMA/ sota/UCMCTrack-master/output/
# cp -r MSTA/outputs/VISO/VISO_MIMMA/ sota/UCMCTrack-master/output/


