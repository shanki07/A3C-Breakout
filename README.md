# A3C-Breakout
* agent.py : Main code to trigger the training , Declare global network and call multiple instances of Worker
  * Code to start training : python agent.py
* Worker.py : Contains codes which call Neural_network.py to build network and calculate loss and apply  gradients to global_network. 
Also saves network weights after every 50,000 runs
* Neural_network.py:Conatains code for building Network and finding Action for state and value function and optimizer code . 
