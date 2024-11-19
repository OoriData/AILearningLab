A collaboration between Oori and [Denver subgroup](https://linktr.ee/denverai) of the [Rocky Mountain AI Interest Group](https://linktr.ee/rmaiig)

We just help people gather to try out hands-on development in GenAI, offering some basic guidance, and expert support during the lab event

# Events

## AI Labs Denver, Nov 2024

[AI Labs - Hands-On Exploration of Agentic AI](https://www.meetup.com/meetup-group-zpqvmxup/events/304518353/?eventOrigin=group_upcoming_events), 19 Nov 2024 at Venture X Denver North

# Attendee set-up

Break up into groups of 2 or 3, usually with one primary host for the GenAIOps. All attendees should ideally bring a laptop, but The GenAIOps host requires one.

Notes on suggested config below. These are in part to align with what the facilitators can most readyly help with, and to reduce the need for any paid or registered third-party services.

**Suggested configs and procedures are just suggestions! Feeel free to use different tools or actions, and then teach your fellow lab participants about your approach!**

We recommend prepping for an AI Lab session by doing the following:

* Revieiwing the configs below to assess which would work best on the laptop you bring
* Review and better yet install any mentioned tools
* Think of some simple, real-life problem you've encountered where GenAI could help

## Case 1: Mac

Primary host on M1/M2/M3/M4 MacBook with at least 8GB RAM, and ideally 16GB

### Suggested approach

Use the [MLX library](https://github.com/ml-explore/mlx) to host local, high performance GenAI models running on GPU (see [their examples](https://github.com/ml-explore/mlx-examples))

### Suggested prep

Install [Python 3.12 from python.org](https://www.python.org/downloads/release/python-3120/), via the "macOS 64-bit universal2 installer"

*Alternatively*: Install Python 3.12 via [Homebrew](https://brew.sh/), but the facilitators will not be able to help as readily with this step

Set up a [virtual environment (venv)](https://medium.com/@KiranMohan27/how-to-create-a-virtual-environment-in-python-be4069ad1efa). A good name for this is `ailab`

Make sure you can run the following command for the venv: `pip install mlx mlx_lm`

## Case 1: Primary host on Windows

TBD

## Case 1: Primary host on Linux

Primary host on a Linux laptop, preferably with a modern GPU, though you can use CPU-only, which will be much slower.

### Suggested approach

Use [Ollama](https://github.com/ollama/ollama)

### Suggested prep

Make sure you have a 3.10 or more recent Python environemnt.

Set up a [virtual environment (venv)](https://medium.com/@KiranMohan27/how-to-create-a-virtual-environment-in-python-be4069ad1efa). A good name for this is `ailab`

DO NOT proceed outside a venv.

Install Ollama:

```sh
curl -fsSL https://ollama.com/install.sh | sh
```

Optionally, also install [Open WebUI](https://github.com/open-webui/open-webui?tab=readme-ov-file#ollama-web-ui-a-user-friendly-web-interface-for-chat-interactions-).

```sh
pip install open-webui
```

Test that you can launch it:

```sh
open-webui serve
```
