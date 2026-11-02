# neural-flow-runtime

Minified browser runtime of the neural-architecture-viz agent skill: lineage-colored
dataflow figures of neural network architectures (tensor tiers, block cards,
breathing flow on the lines). The skill loads it from jsDelivr so that inline
figures in chat stay small:

    <script src="https://cdn.jsdelivr.net/gh/YuXingyao000/neural-flow-runtime@v1.0.1/runtime.min.js"
            integrity="sha384-TfnVIPDsQ1ANu96NincPMGjWMXGwS526bQZpZIS8rukTvJ5I9ZKXGh9Qi4hultKY" crossorigin="anonymous"></script>

`runtime.css` holds the matching styles and `src/` the readable sources it is
minified from. Files under a version tag never change. MIT licensed.
