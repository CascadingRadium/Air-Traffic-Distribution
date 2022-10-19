import { createGlobalState } from "react-hooks-global-state";

const { setGlobalState, useGlobalState } =createGlobalState({
    "value":1
})

export { setGlobalState,useGlobalState };