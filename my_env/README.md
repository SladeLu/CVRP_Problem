# How to use

## Before install:
* make sure you has install:
    1. python3.+ 
    2. pip
* make sure everything in Openai gym
if not ,try
` pip install gym `

## Install pip package my_env
1. open shell or terminal,try
` cd my_env `
` pip install -e .`
2. register my_env to gym env
`cd ...\Python\Python37\Lib\site-packages\gym\envs`
open file __ init __.py
add code blew at the tail of the .py file
`register(
    id='CVRPEnv-v0',
    entry_point='myenv.envs:CVRPEnv'
)`
