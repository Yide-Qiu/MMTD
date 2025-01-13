# python infer_test.py --dataset_name SatSOT --exp_name SatSOT_PMST
python infer_test.py --dataset_name AccessAIS --exp_name AccessAIS_PMST_wo_asso_loss --ckpt_dir AccessAIS_wo_asso_loss
# python infer_test.py --dataset_name MTAD --exp_name MTAD_PMST

cd ..
# cp -r MSTA/outputs/SatSOT/SatSOT_PMST/ sota/UCMCTrack-master/output/
cp -r MSTA/outputs/AccessAIS/AccessAIS_PMST_wo_asso_loss/ sota/UCMCTrack-master/output/
# cp -r MSTA/outputs/SAT_MTB/SAT_MTB_PMST/ sota/UCMCTrack-master/output/






