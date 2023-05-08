# Overview

This is an implementation of the Salesforce [CodeGen](https://github.com/salesforce/CodeGen) model. The model
was trained using the `mono` dataset and this version is the 2B parameter model. This model is specialized for Python
code production.

# Deploying to Baseten

To deploy this Truss on Baseten, first install the Baseten client:

```
$ pip install baseten
```

Then, in a Python shell, you can do the following to have an instance of GFP-GAN deployed
on Baseten:

```python
import baseten
import truss

codegen_mono_handle = truss.load(".")
baseten.deploy(codegen_mono_handle, model_name="Codegen-Mono")
```

## Inputs
The input should be a list of dictionaries must have a key `context` which represents the prompt for generation to the
model. It supports the following keys:
* `prompt` - the natural language or code prompt desired to generate.
* `max_length` - optional, the maximum length for generation, maxes out at 128 tokens
* `temperature` - optional, the temperature for the generator. defaults to 0.2
* `top_p` - optional, the top_p for the generator. defaults to 0.95

For example:

```json
[{
    "prompt": "def fibonacci(n):"
}]
```

## Outputs
The result will be a dictionary that will have the following keys:
* `completion` - the full generation of the model
* `truncation` - a heuristically truncated segment of the code
* `context` - the context provided to the model

For example:

```json
{
    "completion": "code for fibonacci function\r\ndef fib(n):\r\n...",
    "prompt": "code for fibonacci",
    "truncation": " function\r\ndef fib(n):\r\n..."
}
```

## Example

```bash
$ curl -X POST https://app.staging.baseten.co/model_versions/{MODEL_VERSION_ID}/predict -H 'Authorization: Api-Key {YOUR_API_KEY}' -d '{"prompt": "code for fibonacci"}'
{
    "model_id": "8qZKnBg",
    "model_version_id": "vq0582w",
    "model_output": {
        "status": "success",
        "data": {
            "completion": "code for fibonacci function\r\ndef fib(n):\r\n...",
            "prompt": "code for fibonacci",
            "truncation": " function\r\ndef fib(n):\r\n..."}, "message": null
        }
    }
}
```