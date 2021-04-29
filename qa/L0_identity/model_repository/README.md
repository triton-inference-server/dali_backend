# Identity model for DALI

This is a identity model, implemented using DALI.
It passes the input through DALI and returns it unchanged.
This is a CPU version, you can easily modify it to provide GPU output,
by changing `device` parameter for `dali.fn.external_source` operator.

## Setting up the model

To set up the model, you need to serialize DALI pipeline.
`indentity_pipeline.py` shows, how to do it. Just call `serialize` method
on created Pipeline object.

To set up the model repo automatically, you can call `setup_identity_example.sh` script.

## Remember

As always in DALI Backend case, remember that `dali.fn.external_source`'s `name` parameter must match
with input name provided in `config.pbtxt` file. 