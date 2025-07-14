python gpt.py --batch-size 256 --embedding-dim 192 --num-heads 6 --num-layers 4 --tokenizer tinystories_bpe_tokenizer/4K
python gpt.py --batch-size 256 --embedding-dim 384 --num-heads 12 --num-layers 6 --tokenizer tinystories_bpe_tokenizer/8K
python gpt.py --batch-size 128 --embedding-dim 512 --num-heads 16 --num-layers 6 --tokenizer tinystories_bpe_tokenizer/8K
python gpt.py --batch-size 128 --embedding-dim 512 --num-heads 16 --num-layers 12 --tokenizer tinystories_bpe_tokenizer/16K
python gpt.py --batch-size 128 --embedding-dim 512 --num-heads 16 --num-layers 12 --tokenizer tinystories_bpe_tokenizer/32K
python gpt.py --batch-size 128 --embedding-dim 768 --num-heads 12 --num-layers 12 --tokenizer tinystories_bpe_tokenizer/32K
python gpt.py --batch-size 128 --embedding-dim 768 --num-heads 12 --num-layers 16 --tokenizer tinystories_bpe_tokenizer/32K

python gpt.py --batch-size 256 --embedding-dim 192 --num-heads 6 --num-layers 4 --tokenizer tinystories_bpe_tokenizer/4K --num-stories 24000 --max-iterations 4000
python gpt.py --batch-size 256 --embedding-dim 384 --num-heads 12 --num-layers 6 --tokenizer tinystories_bpe_tokenizer/8K --num-stories 24000 --max-iterations 4000
python gpt.py --batch-size 128 --embedding-dim 512 --num-heads 16 --num-layers 6 --tokenizer tinystories_bpe_tokenizer/8K --num-stories 24000 --max-iterations 4000
python gpt.py --batch-size 128 --embedding-dim 512 --num-heads 16 --num-layers 12 --tokenizer tinystories_bpe_tokenizer/16K --num-stories 24000 --max-iterations 4000
python gpt.py --batch-size 128 --embedding-dim 512 --num-heads 16 --num-layers 12 --tokenizer tinystories_bpe_tokenizer/32K --num-stories 24000 --max-iterations 4000
python gpt.py --batch-size 128 --embedding-dim 768 --num-heads 12 --num-layers 12 --tokenizer tinystories_bpe_tokenizer/32K --num-stories 24000 --max-iterations 4000
python gpt.py --batch-size 128 --embedding-dim 768 --num-heads 12 --num-layers 16 --tokenizer tinystories_bpe_tokenizer/32K --num-stories 24000 --max-iterations 4000

python gpt.py --batch-size 384 --embedding-dim 768 --num-heads 12 --num-layers 12 --tokenizer tinystories_bpe_tokenizer/32K \
    --num-stories 48000 --max-iterations 10000 --eval-stories 4096 --eval-interval 500 --learning-rate 1e-4 --context-length 128 --use-torch-compile

python gpt.py --batch-size 256 --embedding-dim 384 --num-heads 12 --num-layers 6 --tokenizer tinystories_bpe_tokenizer/8K \
    --num-stories 48000 --max-iterations 5000 --eval-stories 4096 --eval-interval 250 --learning-rate 1e-4 --context-length 256 --use-torch-compile

python gpt.py --batch-size 256 --embedding-dim 384 --num-heads 12 --num-layers 6 --tokenizer tinystories_bpe_tokenizer/8K \
    --num-stories 48000 --max-iterations 0 --eval-stories 4096 --eval-interval 250 --learning-rate 1e-4 --context-length 256 --use-torch-compile \
    --load models/model_384d_6l_12h_tinystories_bpe_tokenizer_8K_20250713_142458.pth


python gpt.py --batch-size 256 --embedding-dim 128 --num-heads 4 --num-layers 8 --tokenizer tinystories_bpe_tokenizer/4K

# quick

# 1M
python gpt.py --batch-size 32 --embedding-dim 96 --num-heads 3 --num-layers 6 --tokenizer tinystories_bpe_tokenizer/2K \
    --num-stories 24000 --max-iterations 5000 --eval-stories 4096 --eval-interval 500 --learning-rate 5e-4 --context-length 128 --use-torch-compile

# 2.5M
python gpt.py --batch-size 512 --embedding-dim 128 --num-heads 4 --num-layers 8 --tokenizer tinystories_bpe_tokenizer/4K \
    --num-stories 48000 --max-iterations 10000 --eval-stories 4096 --eval-interval 500 --learning-rate 5e-4 --context-length 128 --use-torch-compile

# full

# 2.5M
python gpt.py --batch-size 512 --embedding-dim 128 --num-heads 4 --num-layers 8 --tokenizer tinystories_bpe_tokenizer/4K \
    --num-stories 96000 --max-iterations 50000 --eval-stories 4096 --eval-interval 500 --learning-rate 5e-4 --context-length 128 --use-torch-compile
