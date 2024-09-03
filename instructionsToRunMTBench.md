

```
cd mtbencheval/FastChat
pip install -e ".[model_worker,llm_judge]"
```
(To minimize env distruptions, you can first openup the pyroject.toml and see what all's gonna be installed. Install the stuff yourself if you can one by one so  that
you dont have suddenly a long chain of dependencies with something getting installed. This will ensure that the final pip install -e . only installs 1-2 new packages apart
from the repo building itself, which will make your job easy. Nevertheless keep a close eye out for the last command too)

```
python download_mt_bench_pregenerated.py
```


```
python gen_model_answer.py --model-path lmsys/vicuna-7b-v1.5 --model-id vicuna-7b-v1.5
```

Ensure OpenAI_API_KEY is set.

Now run judgement on the thing you generated above [it may ask you to cluck yes after you run it.

```
python gen_judgment.py --model-list vicuna-7b-v1.5 --parallel 2
```

You can see the results via
```
python show_result.py --model-list vicuna-7b-v1.5
```
You get 2 turn-level scores and 1 overall aggregate. For vicuna its like 6.11

FAQs and declutter section:


Why do we need most recent repo vrsion?
LLama3 related fixes were added only recently.
e.g. see https://github.com/lm-sys/FastChat/pull/3326


What runs so far and checked?
Huggingface lmsys vicuna
Llama3 [will check next]
llama3 with peft adapter [will check after]
