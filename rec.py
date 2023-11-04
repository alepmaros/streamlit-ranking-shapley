import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, JsCode

import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

from streamlit_shap import st_shap
import shap

st.text_input("Seed", value=1, key="seed")

if st.button("Send request", key="send_request"):
    data = load_diabetes()

    df = pd.DataFrame(data["data"], columns=data["feature_names"])
    df["target"] = data["target"]

    X_sample = df.sample(5, random_state=int(st.session_state.seed)).drop("target", axis=1)


    df = df.drop(X_sample.index)

    y = df["target"]
    X = df.drop("target", axis=1)

    model = RandomForestRegressor(random_state=int(st.session_state.seed))
    model.fit(X, y)


    X_sample["prediction"] = model.predict(X_sample)

    st.write("Prediction dataframe:")

    render_image = JsCode("""
    class ThumbnailRenderer {
        init(params) {
            this.eGui = document.createElement('img');
            this.eGui.setAttribute('src', params.value);
            this.eGui.setAttribute('width', '50');
            this.eGui.setAttribute('height', '50');
        }
        getGui() {
            return this.eGui;
        }
    }
    """)


    X_sample["photo"] = "https://i.imgur.com/MYmm7E1.jpeg"
    cols = X_sample.columns.to_list()
    cols = cols[-2:] + cols[:-2]
    X_sample = X_sample[cols]

    gb = GridOptionsBuilder.from_dataframe(X_sample)
    gb.configure_auto_height(autoHeight=True)
    gb.configure_grid_options(rowHeight=50)
    gb.configure_selection("single", use_checkbox=True)
    gb.configure_column("photo", cellRenderer=render_image)
    gridOptions = gb.build()

    grid_response = AgGrid(
        X_sample,
        gridOptions=gridOptions,
        data_return_mode=DataReturnMode.AS_INPUT,
        allow_unsafe_jscode=True,
        theme='streamlit'
    )

    selected = grid_response['selected_rows']

    Xplainer = X_sample.drop(["prediction", "photo"], axis=1)
    explainer = shap.Explainer(model.predict, Xplainer)
    shap_values = explainer(Xplainer)

    print(shap_values)

    if selected:
        st_shap(shap.plots.waterfall(shap_values[int(selected[0]["_selectedRowNodeInfo"]["nodeId"])]), width=700)

    st_shap(shap.plots.beeswarm(shap_values), width=700)