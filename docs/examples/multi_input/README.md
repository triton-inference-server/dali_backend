# Multi input model for DALI Backend

This is a multi input model for DALI preprocessing.
It passes multiple inputs through DALI and returns them unchanged.
The inputs are both CPU and GPU, see the `device` parameter in `dali.fn.external_source` operator.

## Setting up the model

To set up the model, you need to serialize DALI pipeline.
`multi_input_pipeline.py` shows how to do it. Just call `serialize` method
on created Pipeline object.

To set up the model repo automatically, you can call `setup_multi_input_example.sh` script.

## Remember

As always in DALI Backend case, remember that `dali.fn.external_source`'s `name` parameter must match
with input name provided in `config.pbtxt` file. 
