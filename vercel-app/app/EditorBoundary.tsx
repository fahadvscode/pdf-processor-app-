"use client";
import { Component, type ReactNode } from "react";

type Props = { onClose: () => void; children: ReactNode };
type State = { error: string };

/** Keeps a failure inside the white-out window from taking down the page. */
export default class EditorBoundary extends Component<Props, State> {
  state: State = { error: "" };

  static getDerivedStateFromError(err: unknown): State {
    return { error: err instanceof Error && err.message ? err.message : "Something went wrong" };
  }

  componentDidCatch(err: unknown) {
    console.error("White-out editor failed", err);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="wo-backdrop">
        <div className="wo-modal wo-modal-error" role="alertdialog">
          <h3>Couldn&apos;t open the white-out editor</h3>
          <p className="caption">{this.state.error}</p>
          <p>Your file is still in the upload list. You can upload it without white-out or try again.</p>
          <button
            className="btn btn-primary"
            onClick={() => {
              this.setState({ error: "" });
              this.props.onClose();
            }}
          >
            Close
          </button>
        </div>
      </div>
    );
  }
}
