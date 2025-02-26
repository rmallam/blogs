Unknown command: -m
Available commands: 
  --run (-r): Run a model previously converted into ggml
              ex: -m /models/7B/ggml-model-q4_0.bin -p "Building a website can be done in 10 simple steps:" -n 512
  --bench (-b): Benchmark the performance of the inference for various parameters.
              ex: -m model.gguf
  --perplexity (-p): Measure the perplexity of a model over a given text.
              ex: -m model.gguf -f file.txt
  --convert (-c): Convert a llama model into ggml
              ex: --outtype f16 "/models/7B/" 
  --quantize (-q): Optimize with quantization process ggml
              ex: "/models/7B/ggml-model-f16.bin" "/models/7B/ggml-model-q4_0.bin" 2
  --all-in-one (-a): Execute --convert & --quantize
              ex: "/models/" 7B
  --server (-s): Run a model on the server
              ex: -m /models/7B/ggml-model-q4_0.bin -c 2048 -ngl 43 -mg 1 --port 8080
