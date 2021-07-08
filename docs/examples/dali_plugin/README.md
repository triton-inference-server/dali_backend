# DALI plugins with Triton

This example shows how to use custom operations implemented for DALI
(i.e. DALI plugins) with Triton. In this example, we use a custom DALI
operation that copies the input to the output. You can find this copy plugin in
[DALI documentation](https://docs.nvidia.com/deeplearning/dali/master-user-guide/docs/examples/custom_operations/custom_operator/create_a_custom_operator.html).

### Step 1: Obtain the plugin lib

Firstly, you need to code and build the plugin. This topic is
out of scope of this tutorial. You can find all relevant information in
[DALI documentation](https://docs.nvidia.com/deeplearning/dali/master-user-guide/docs/examples/custom_operations/custom_operator/create_a_custom_operator.html).
All you need to do is to follow the linked tutorial to obtain`.so` library with your custom plugin.

### Step 2: Serialize the DALI pipeline which uses your custom operation

As always when using DALI with Triton, you need to serialize your DALI pipeline.
Again, this topic is broadly discussed in the tutorial linked above. To summarize,
you need to use `nvidia.dali.plugin_manager` to load your plugin and use it in
your pipeline. `custom_copy_pipeline.py` file shows, how do to it:

    plugin_manager.load_library('/path/to/libcustomcopy.so')
    
    @dali.pipeline_def(batch_size=3, num_threads=1, device_id=0)
    def pipe():
        data = dali.fn.external_source(device="cpu", name="DALI_INPUT_0")
        cp = dali.fn.custom_copy(data)
        return cp

### Step 3: Inject the plugin lib into tritonserver docker image

The `.so` file you built in the previous step contains the
definition of your operation. Therefore, you need to inject it
into your tritonserver image, so that DALI would be able to conduct
this operation. There are multiple ways to achieve that, including
mounting (`-v` option) or copying (`COPY` directive or `docker cp` command).

Here we will assume, that your `.so` library is located at
`/models/libcustomcopy.so` in your tritonserver docker image.

### Step 4: Run `tritonserver` with DALI plugin

For DALI Backend to be able to use your plugin libraries, you must specify their paths
when running the server. You shall use a backend configuration for this. Please add a
`plugin_libs` parameter to the DALI Backend configuration (`--backend-config`), with the value being a
colon-separated (or semicolon-separated on Windows) list of paths to plugin libraries, e.g.:

    $ docker run <your typical docker args> nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/models --backend-config dali,plugin_libs=/models/libcustomcopy.so:/path/to/libplugin2.so

In this example, DALI Backend will try to load two plugin libaries: `/models/libcustomcopy.so` and
`/path/to/libplugin2.so`.

### Step 5: Enjoy your custom DALI plugin in Triton

That's all. You should have the server running by now.

## Common errors
Let's assume you would like to use `CustomCopy` operator. If at the point of server
initialization you encounter an error:

    Error when adding operator CustomCopy: Schema for operator 'CustomCopy' not registered

This probably means, that the plugin `.so` library failed to load. You might want to
double-check your backend configuration and if the plugin library is available in the location
you specified.

## Remember

As always in DALI Backend case, remember that `dali.fn.external_source`'s `name` parameter must match
the input name provided in `config.pbtxt` file. 