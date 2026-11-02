
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="ML Drift Monitoring Lab",
    page_icon="📊",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

st.title("ML Drift Monitoring Lab")
st.caption(
    "Exploring model degradation, feature drift, "
    "and drift detector reliability."
)


@st.cache_data
def load_csv(filename):
    path = RESULTS / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


monitoring = load_csv("electricity_monitoring.csv")
detectors = load_csv("detector_comparison.csv")
validation = load_csv("temporal_validation.csv")
predictions = load_csv("temporal_predictions.csv")


tab1, tab2, tab3 = st.tabs([
    "Model Health",
    "Drift Detectors",
    "Model Comparison"
])


with tab1:
    st.header("Real-World Model Monitoring")

    if monitoring is None:
        st.info(
            "Run the export cell in Notebook 05 to "
            "generate electricity_monitoring.csv."
        )
    else:
        st.write(
            "A frozen logistic regression model monitored "
            "on chronological electricity-market data."
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Mean batch accuracy",
            f"{monitoring['accuracy'].mean():.1%}"
        )
        c2.metric(
            "Mean batch ROC-AUC",
            f"{monitoring['roc_auc'].mean():.3f}"
        )
        c3.metric(
            "Monitoring batches",
            len(monitoring)
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monitoring["batch"],
            y=monitoring["accuracy"],
            name="Accuracy"
        ))
        fig.add_trace(go.Scatter(
            x=monitoring["batch"],
            y=monitoring["roc_auc"],
            name="ROC-AUC"
        ))
        fig.update_layout(
            title="Model Performance Over Time",
            xaxis_title="Batch",
            yaxis_title="Score",
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feature Drift — NSW Demand")

        psi_threshold = st.slider(
            "PSI alert threshold",
            min_value=0.1,
            max_value=2.0,
            value=0.2,
            step=0.1
        )

        fig = px.line(
            monitoring,
            x="batch",
            y="psi",
            title="Population Stability Index"
        )
        fig.add_hline(
            y=psi_threshold,
            line_dash="dash",
            annotation_text="Selected threshold"
        )
        st.plotly_chart(fig, use_container_width=True)

        alert_count = int(
            (monitoring["psi"] > psi_threshold).sum()
        )
        st.write(
            f"**{alert_count} of {len(monitoring)} batches** "
            "exceed the selected PSI threshold."
        )

        st.caption(
            "A PSI exceedance indicates distribution change, "
            "not necessarily model degradation."
        )


with tab2:
    st.header("Drift Detector Reliability")

    if detectors is None:
        st.warning("detector_comparison.csv not found.")
    else:
        st.write(
            "Synthetic benchmark across stable, feature-drift, "
            "and concept-drift scenarios."
        )

        metric = st.selectbox(
            "Comparison metric",
            ["false_alert_rate", "detection_rate",
             "mean_detection_delay"]
        )

        fig = px.bar(
            detectors,
            x="detector",
            y=metric,
            color="scenario",
            barmode="group",
            title=metric.replace("_", " ").title()
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            detectors.round(3),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "Detection rates are exploratory. Repeated testing "
            "can cause chance alerts, and label-based detectors "
            "cannot act before ground-truth labels arrive."
        )


with tab3:
    st.header("Temporal Model Validation")

    if validation is None:
        st.warning("temporal_validation.csv not found.")
    else:
        st.write(
            "Early and recent logistic regression models "
            "evaluated on the same chronological test period "
            "using lagged features."
        )

        cols = st.columns(3)

        for col, metric in zip(
            cols, ["accuracy", "roc_auc", "log_loss"]
        ):
            fig = px.bar(
                validation,
                x="model",
                y=metric,
                color="model",
                title=metric.replace("_", " ").title(),
                text_auto=".3f"
            )
            fig.update_layout(showlegend=False)
            col.plotly_chart(fig, use_container_width=True)

        if predictions is not None:
            st.subheader("Prediction Performance Over Time")

            window = st.slider(
                "Rolling window (observations)",
                min_value=100,
                max_value=2000,
                value=500,
                step=100
            )

            chart_data = predictions.copy()

            for name in ["early", "recent"]:
                predicted = (
                    chart_data[f"{name}_probability"] >= 0.5
                ).astype(int)

                correct = (
                    predicted == chart_data["actual"]
                ).astype(float)

                chart_data[f"{name}_rolling_accuracy"] = (
                    correct.rolling(
                        window, min_periods=window
                    ).mean()
                )

            fig = px.line(
                chart_data,
                x="observation",
                y=[
                    "early_rolling_accuracy",
                    "recent_rolling_accuracy"
                ],
                title="Rolling Accuracy on Future Observations"
            )
            fig.update_layout(
                yaxis_title="Rolling accuracy"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Later training data outperformed the early model "
            "in this experiment; this does not isolate the "
            "causal effect of retraining."
        )


st.divider()
st.caption(
    "Educational portfolio project using public electricity "
    "data and controlled synthetic experiments."
)