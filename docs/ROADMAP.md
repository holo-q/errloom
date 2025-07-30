# TODOs

@argp.py
* support 'uv run main <loom/holoware> new <session_name>'
* support 'uv run main <loom/holoware> vllm'
* support 'uv run main <loom/holoware> resume'
* support '--openrouter'

@main.py
* prompt the user for their keys when using --openai, --openrouter, etc. and store them into ~/.erloom (see @paths.py for app standards) as securely as possible


# TODO: Logging alignment

Run this

errloom  🍣 main 🏎️ 💨 ×5🐍 v3.13.5 
❮ uv run main compressor.hol

and this


errloom  🍣 main 🏎️ 💨 ×5🐍 v3.13.5 
❮ uv run main compressor.hol dry

To see the logging standard we're aiming for.

Then try 

errloom  🍣 main 🏎️ 💨 ×5🐍 v3.13.5 
❮ uv run main compressor.hol train --cpu --test_steps 1 --n 1

you will see that the logging is not quite to the same aesthetic. It's more 'spiky'