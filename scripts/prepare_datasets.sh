set -e

TARGET_DIR="./datasets/art"
mkdir -p $TARGET_DIR/train
mkdir -p $TARGET_DIR/train
for VARIABLE in "monet2photo" "cezanne2photo" "ukiyoe2photo" "vangogh2photo"
do
	bash scripts/download_datasets.sh $VARIABLE
	mv scripts/datasets/$VARIABLE/trainA $TARGET_DIR/train/$VARIABLE
    mv scripts/datasets/$VARIABLE/testA $TARGET_DIR/test/$VARIABLE
    # copy photos directory only once
    if $VARIABLE = "monet2photo"
    then
        mv scripts/datasets/$VARIABLE/trainB $TARGET_DIR/train/photos
        mv scripts/datasets/$VARIABLE/testB $TARGET_DIR/test/photos
    fi
    rm -r scripts/datasets/$VARIABLE
done
rm -r scripts/datasets