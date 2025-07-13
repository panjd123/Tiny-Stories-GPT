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

python gpt.py --batch-size 128 --embedding-dim 768 --num-heads 12 --num-layers 12 --tokenizer tinystories_bpe_tokenizer/32K \
    --num-stories 48000 --max-iterations 10000 --eval-interval 500 --learning-rate 1e-4 --context-length 128 --use-torch-compile
