mkdir checkpoints
cd checkpoints
rm *
wget 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-512x512.pkl'
mv stylegan2-ffhq-512x512.pkl faces.pkl
curl -o cats.pkl https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl 
curl -o lions.pkl https://storage.googleapis.com/self-distilled-stylegan/lions_512_pytorch.pkl 
curl -o dogs.pkl https://storage.googleapis.com/self-distilled-stylegan/dogs_1024_pytorch.pkl 
curl -o horses.pkl https://storage.googleapis.com/self-distilled-stylegan/horses_256_pytorch.pkl 
curl -o elephants.pkl https://storage.googleapis.com/self-distilled-stylegan/elephants_512_pytorch.pkl 
curl -o bicycles.pkl https://storage.googleapis.com/self-distilled-stylegan/bicycles_256_pytorch.pkl 
curl -o giraffes.pkl https://storage.googleapis.com/self-distilled-stylegan/giraffes_512_pytorch.pkl 
curl -o cars.pkl http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-car-config-f.pkl 
curl -o churches.pkl http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-church-config-f.pkl 
curl -o metfaces.pkl https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl 
