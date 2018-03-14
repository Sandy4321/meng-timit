if [ ! $# -eq 4 ]; then
    echo "Need to provide new recipe name, (existing) recipe name for dirty data, an absolute path to the feature directory (with subdirs train/, dev/ and test/) and the name of the SCP files that will used for the recipe"
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

clean_recipe_name=single_rir_baseline_clean # Assuming we always use TIMIT clean feats
if [ ! -d $RECIPES/$clean_recipe_name ]; then
    echo "Clean recipe $clean_recipe_name does not exist; exiting"
    exit 1
fi

# Verify features
feats_dir=$3
if [ ! -d $feats_dir ]; then
    echo "Feature dir does not exist at path $feats_dir"
    exit 1
fi

scp_name=$4
for subdir in train dev test; do
    if [ ! -f $feats_dir/$subdir/$scp_name ]; then
        echo "Feature SCP does not exist at path $feats_dir/$subdir/$scp_name"
        exit 1
    fi
done

# Generate template
cd recipes
./generate.sh $new_recipe_name augmented || exit 1

# Point to correct recipes
cd $new_recipe_name
printf "export CLEAN_RECIPE=$RECIPES/$clean_recipe_name\nexport DIRTY_RECIPE=$RECIPES/$dirty_recipe_name\n" > data_config.sh

printf "Using data from $feats_dir\n" > README.txt

# Copy in our features
for subdir in train dev test; do
    # Copy features, taking advantage of Kaldi's compression
    echo "Copying $subdir features..."
    output_dir=$RECIPES/$new_recipe_name/data/$subdir
    mkdir -p $output_dir
    copy-feats --compress=true scp:$feats_dir/$subdir/$scp_name ark,scp:$output_dir/feats.ark,$output_dir/feats.scp || exit 1
done

cd ~/meng-timit
echo "Done setting up recipe!"
