# Identity model for DALI

This is a identity model, implemented using DALI.
It passes the input through DALI and returns it unchanged.
This is a CPU version, you can easily modify it to provide GPU output,
by changing `device` parameter for `dali.fn.external_source` operator.

## Setting up the model

This example does not need any setting up.

## Remember

As always in DALI Backend case, remember that `dali.fn.external_source`'s `name` parameter must match
with input name provided in `config.pbtxt` file. 