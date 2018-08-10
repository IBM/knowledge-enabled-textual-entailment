mkdir ../data/embeddings
cd ../data/embeddings

# download glove
mkdir glove
cd glove
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
# compress glove
gzip glove.840B.300d.txt
rm glove.840B.300d.zip
cd ..

# download conceptnet ppmi
mkdir conceptnet
cd conceptnet
wget https://conceptnet.s3.amazonaws.com/precomputed-data/2016/numberbatch/16.09/conceptnet-55-ppmi.h5
