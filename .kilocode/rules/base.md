# General Context

Errloom is a fork of Verifiers and we are renaming it into a standalone new project.
It's a swiss army knife for RL-enhanced prompt scaffolding, built on new LLM native abstractions like the loom, tapestry, etc.
The main difference from Verifiers is that the trainer doesn't know anything about the dataset anymore. The loom's weaving instead
is sampling rows from the training dataset which is passed to the rollout, and the rollout is able to inject these in any order of its
choosing into the context. @tapestry.py and @holophore.py act as mutable state structs and make abstractions as much as possible of the
chat/completion API. The context must explicitly track mask positions and map these when turning into tokens to decide which tokens should
be included and which masked out. If you see references to a 'prompt' during training this is the part that we are abolishing and consolidating.

# Coding Standard

- Never use kwargs for anything but delegation.
- Never use hasattr or getattr unless explicitly recommended by the user. A field exists or it doesn't.
- Watch out for functional code patterns. We write the code more like Rust
  or C# with usage of mutable structs as interchangeable and intercomposeable
  blocks of work states. We consider that functional coding patterns are a
  mind virus that has ruined the productivity and quality of modern engineering.

We must settle on a sane usage pattern of the code, and optimize and enforce it.
We don't want to write code that can cover infinite possible use-case. We encode the
domain space into the data structures and add extension points on top in the future
when necessary. Refactoring is not seen as a thing to avoid, but as a continuous
elementary part of work during implementation.

# Pair Programming Standard

When working with the user manager, do not be afraid to ask questions,
or question the existing code and whether it meets the requirements
or makes your work harder. Never end your replies without proactively proposing next steps,
revealing the teleology mesh of the work domain to continue traveling latent space
further towards perfection. You act as the butler to all possibility, and you
must ensure that the possibility-space is optimally reviewed by the user at each
to ensure adherance to the target domain problem and requirements.


# Work Standard

Enumerate the ambiguities and possible interpretations, choices, and considerations to make regarding
implementation, and communicate efficiently back and forth to avoid misunderstandings.
We are proactive about truth-seeking towards the ultimate state of the program, so
we are not afraid to constantly and continuously review whether the backend architecture
should be upgraded to facilitate a new usage-pattern on the API user end.
It's better to exchange a few small messages etching the exact vision first before a task
or implementation to ensure that we won't need to revert code or make more changes afterwards.
Use efficient language and symbolism to relay a vision, no need to get too verbose.
We can jack in and lock in together into spontaneous "dsl" of communication so to speak.

When you are actively working on an implementation or a bug and you resolve it,
always ASK before moving to a different task, bug, or idea. PROPOSE instead what
are the next options. Absolutely do not under any circumstance jump to a different
task or work item. Think in terms of 'tickets' - "would I be jumping into a different ticket" ?

Make a conscious effort to harmonize with the existing coding style, naming conventions, logging patterns, ...
Rather than the way you would normally do it. Always aim to perceive and model the soul of the developer behind the code
as you work, to optimize against this soul rather than the surface objective.

# Definitions

- Holoware: a prompting program.
- Holophore: an holowaric emmanation. (final context trace)
- Holotypes: a class created specifically to implement __holo__ and be used in Holoware.
- Holofunc: a __holo__ class method.


# Shared Pains

We define here the observations we have made so that we operate on the same wavelength

- The distinctions between chat and completion API is finicky and overly complicates creative use of LLMs. We optimize our wrappers to prevent this.
- We discourage magic dictionaries heavily. We always create dataclasses, and sometimes prefer pydantic models for serialization and communication domains.
- We employ a strong logging standard to maximize spatial awareness over the code as it is executing, laying out all the of the important data to understand the execution parameters to constrain the debugging bounds.
- We come from C# and write the code to this standard. Private access, properties that clean up the code, strong typing, etc. Don't encapsulate unnecessarily, only things that clean up the code. this isn't java either.

# When writing any logging

- We use the logging and rich libraries.
- Consider the larger flow of the application and aim to encapsulate the full program trace, every large movement.
- Consider the final logs from a standpoint of symbolism, visual patterns, and semiotic rhythm, to avoid overly verbose logs.
- Logs as part of a loop flow should be simulated in your mind to verify the final visual block has emergent flow and structure, aesthetic.
- Maximize the use of rich colors and text formatting for increase. Use other formatting constructs like Box and headers where it makes sense, but considering the larger flow
- Maintain a log_design.md at the root of the workspace which contains an abridged demonstration or simulation of the program flow based on your logging, incorporating sections and using [...] to highlight and compare possible juxtapositions.
- Be proactive about replacing and improving the logging.

This ensures that the final program output will always be fantastic and easy to parse at a glance.

The style should be structured, improvised DSL-like, clean, easily parsed, minimal and non-verbose, max signals & information for least visual noise.

# Commands

Use uv for commands (uv run, uv pip, uv run pytest ...)

# run

Home screen
> uv run main

Dry test to visualize how the holoware is running, using MockClient that returns MOCK instead of calling vllm
> uv run main hol/compressor.hol dry

Append --debug if you want to see down to debug log level.

## test

> uv run testslide <path>

Some useful args with testslide:

--filter-text FILTER_TEXT
Only execute examples that include given text in their names (test functions, doesn't apply to TestCase classes)

--fail-fast
Stop execution when an example fails.

from tests.base import ErrloomTest

Instead of TestCase.
It handles some stuff with our logging architecture.

## commit

Never commit work by yourself without asking first.

Never make a commit without checking git status first.
There may be other stuff than what we worked on.
Add everything and check all the diffs to write
a comprehensive message.

