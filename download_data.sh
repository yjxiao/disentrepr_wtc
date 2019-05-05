mkdir data
cd data

echo "Downloading dSprites dataset."
if [[ ! -d "dsprites" ]]; then
  mkdir dsprites
  wget -O dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
fi
echo "Downloading dSprites completed!"

echo "Downloading cars dataset."
if [[ ! -d "cars" ]]; then
  wget -O nips2015-analogy-data.tar.gz http://www.scottreed.info/files/nips2015-analogy-data.tar.gz
  tar xzf nips2015-analogy-data.tar.gz
  rm nips2015-analogy-data.tar.gz
  mv data/cars .
  rm -r data
fi
echo "Downloading cars completed!"
