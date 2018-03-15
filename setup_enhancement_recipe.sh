if [ ! $# -eq 3 ]; then
    echo "Need to provide new recipe name, (existing) recipe name for dirty data and an absolute path to the feature directory (with subdirs train/, dev/ and test/)"
    exit 1
fi

. ./path.sh
echo "Setting up environment..."
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/data/sls/u/meng/skanda/cuda/lib64:$LD_LIBRARY_PATH
source activate $KALDI_ENV
echo "Environment set up."

# Set up recipe names
new_recipe_name=$1
if [ -d $RECIPES/$new_recipe_name ]; then
    echo "Recipe name $new_recipe_name already exists in $RECIPES; exiting"
    exit 1
fi

dirty_recipe_name=$2
if [ ! -d $RECIPES/$dirty_recipe_name ]; then
    echo "Dirty recipe $dirty_recipe_name does not exist; exiting"
    exit 1
fi

# Verify features
feats_dir=$3
if [ ! -d $feats_dir ]; then
    echo "Feature dir does not exist at path $feats_dir"
    exit 1
fi

for subdir in train dev test; do
    if [ ! -d $feats_dir/$subdir ]; then
        echo "Subdir $subdir does not exist at path $feats_dir"
        exit 1
    fi

    for scp_name in src_clean-tar_clean.scp src_dirty-tar_clean.scp; do 
        if [ ! -f $feats_dir/$subdir/$scp_name ]; then
            echo "Feature SCP $scp_name does not exist at path $feats_dir/$subdir"
            exit 1
        fi
    done
done

# Generate template
cd recipes
./generate.sh $new_recipe_name enhancement || exit 1

# Point to correct recipes
cd $new_recipe_name
printf "export DIRTY_RECIPE=$RECIPES/$dirty_recipe_name\n" > data_config.sh

printf "Using data from $feats_dir\n" > README.txt

# Copy in our features, taking advantage of Kaldi's compression
for subdir in train dev test; do
    # Copy reconstructed clean features
    scp_name=src_clean-tar_clean.scp
    echo "Copying $subdir/$scp_name..."
    output_dir=$RECIPES/$new_recipe_name/reconstructed_data/$subdir
    mkdir -p $output_dir
    copy-feats --compress=true scp:$feats_dir/$subdir/$scp_name ark,scp:$output_dir/feats.ark,$output_dir/feats.scp || exit 1
    
    # Copy enhanced dirty features
    scp_name=src_dirty-tar_clean.scp
    echo "Copying $subdir/$scp_name..."
    output_dir=$RECIPES/$new_recipe_name/enhanced_data/$subdir
    mkdir -p $output_dir
    copy-feats --compress=true scp:$feats_dir/$subdir/$scp_name ark,scp:$output_dir/feats.ark,$output_dir/feats.scp || exit 1
done

cd ~/meng-timit
echo "Done setting up recipe!"
