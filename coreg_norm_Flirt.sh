#!/bin/bash
#example to run coreg_norm_Flirt.sh /gpfs/hpchome/gagand87/project/ highres001_brain.nii.gz MNI152_T1_2mm_brain vol1Flirt vol1Flirt.nii.gz
export FSL_DIR='/gpfs/hpchome/gagand87/project/fsl'
export out_dir=$1
#echo $1 $2 $3 $4 $5

#$1=out_dir $2=high res, $3 = ref $4=nifti in $5 = nifti out
#output file name w8 ext
export o_name=`echo $5|awk -F'.' '{print $1}'`
#output matrix name
export om_n1=`echo $o_name|sed 's/$/_1.mat/g'`
export om_n2=`echo $o_name|sed 's/$/_2.mat/g'`
#final output matrix name
export om_n=`echo $o_name|sed 's/$/.mat/g'`

#echo $om_n1 $om_n2 $om_n $o_name

echo "$FSL_DIR/bin/flirt -in $2 -ref $3 -omat $out_dir/$om_n1 -bins 512 -cost mutualinfo -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -dof 12"
echo
$FSL_DIR/bin/flirt -in $2 -ref $3 -omat $out_dir/$om_n1 -bins 512 -cost mutualinfo -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -dof 12

echo "$FSL_DIR/bin/flirt -in $4 -ref $2 -omat $out_dir/$om_n2 -bins 512 -cost mutualinfo -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -dof 12"
echo
$FSL_DIR/bin/flirt -in $4 -ref $2 -omat $out_dir/$om_n2 -bins 512 -cost mutualinfo -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -dof 12

echo "$FSL_DIR/bin/convert_xfm -concat $out_dir/$om_n1 -omat $out_dir/$om_n $out_dir/$om_n2"
echo
$FSL_DIR/bin/convert_xfm -concat $out_dir/$om_n1 -omat $out_dir/$om_n $out_dir/$om_n2

echo "$FSL_DIR/bin/flirt -in $4 -ref $3 -out $out_dir/$5 -applyxfm -init $out_dir/$om_n -interp spline"
echo
$FSL_DIR/bin/flirt -in $4 -ref $3 -out $out_dir/$5 -applyxfm -init $out_dir/$om_n -interp spline

echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"