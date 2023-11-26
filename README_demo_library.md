#  Distributed Fine-Tuning of Language Models (LLMs)

**Expected Demo Time: 15min**

This demo demonstrates how to perform fine-tuning of large language models (LLM) in
a distributed manner.

## When to use this demo?

The customer:
* interested in creating LLM with their data
* realizes that running LLM on one machine is slow and costly
* improving language model performance for specific tasks is a priority

### When NOT to use this demo?
This is a long and complex demo. Don't prioritise this demo unless the customer's has a strong engineering focus and
a highly interested in training LLMs in scalable, distributed manner.


## Video


## How to demo?

> We have an existing project that's connected to the right Git repository and allows you to demo this easily.

### Before the demo:
1. Create a new Valohai project and connect it the [valohai-llms](https://github.com/SofiaChar/valohai-distributed-llms) repository.
   * You can also use [this project]
2. The repo consists of three main scripts that enable distributed training:
    * Distributed training within one machine (should have more than 1 GPU):
      * **Torchrun - the feature from Pytorch**. 
        * Doesn't really need any changes to the regular training script.
        * In our case we use transformers.Trainer - distribution is managed fully by torchrun.
        * To run we use `torchrun --nproc-per-node=4 train-torchrun.py`
      * **Accelerate - the tool from Hugging Face - Accelerator**.
        * Requires some changes to the regular training script.
        * We use the regular training loop.
        * To run we use `python -m accelerate.commands.launch --num_processes=4 train-accelerator.py`
    * **Distributed training between few machines (one GPU per each is fine)**:
      * Uses valohai.distributed feature. Of course some extra code is required.
      * We have to do data partition (divide the dataset into parts, subset per machine)
      * We calculate average gradients (between machines) after each step of the training.
      * `python train-task.py`

We are fine-tuning **Bert Large CNN** model with summarization dataset from Samsung called **SAMSum**.
### Demo :popcorn:

For the demo we need `p3.8xlarge` machine, sometimes it takes time to get it, so better to have it ready when the demo starts. 

#### Distribute within one machine:
1. Create execution `train-accelerator`. Show that in the command we have `--num_processes=4`, which means that the work will be distributed between all 4 GPUs of the machine.
2. Wait until everything is prepared and the training have started: (9%|▊ | 343/3683 [02:40<**15:55**). Show that the whole training is going to take 15 minutes.
3. Stop this execution, and copy this execution change only one parameter in command `--num_processes=2`.
4. After everything is prepared, you will see that now it will take 2 times longer to fine-tune. (9%|▊ | 643/7366 [02:40<**27:55**)

#### Distribute between few machines:
1. Create Task using the `train-task`. Show that now we use cheaper machine with one GPU -`g4dn.xlarge`.
2. Choose Task type - Distributed. **Execution count 2**. Run the Task.
3. Go to execution and show: 
    * `Announcing distributed group membership, I am 1. All 2/2 members announced themselves in 0.00 seconds...`
4. Show that **737 steps** are need to finish the fine-tuning.
5. Create another Task set **execution count to 4**. 
6. Show that now we have 4 members
   * `3/4 members in distributed group after 2.23 seconds... All 4/4 members announced themselves ...`
7. Show that **368 steps** are need to finish the fine-tuning. Because we do the data partition, the number of steps depends on the number of machines.


TO DO:
- Add points to show same valohai features?