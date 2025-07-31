# TODOS: Local features

@argp.py
* support 'uv run main <loom/holoware> new <session_name>'
* support 'uv run main <loom/holoware> vllm' which runs vllm_server cli, just like the vllm launch script in @pyproject.toml
* support 'uv run main <loom/holoware> resume'
* support '--openrouter'


@main.py
* prompt the user for their keys when using --openai, --openrouter, etc. and store them into ~/.erloom (see @paths.py for app standards) as securely as possible

@log.py @argp.py
* add an argument to reset the persisted column width
* track the longest column width for this session separately and add a function to save it to persistence +10%, and call the function at the end of execution in @main.py in the most reliable way possible. this makes the persistence contextually local rather than infinitely spanning

# TODOS

* will added many new environments https://github.com/willccbb/verifiers/tree/main/environments lets rip them off into holowares to see how our api handles stuff
*  


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

you will see that the logging is not quite to the same aesthetic. It's more 'spiky'.
_prepare_inputs should 

# TODO: JetBrains Holoware syntax

we already have the vscode one so it shouldn't be too hard
however it may not be as simple to create a plugin and install it locally
with vs code we can simply symlink the plugin directory and it's all json and textmate stuff
with jetbrains there may be some jar bullshit and we may not be able to test it without some convoluted setup
i dont know