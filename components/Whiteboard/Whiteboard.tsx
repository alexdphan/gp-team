import { Tooltip } from "../../primitives/Tooltip";
import { useBoundingClientRectRef } from "../../utils";
import { Cursors } from "../Cursors";
import { WhiteboardNote } from "./WhiteboardNote";
import styles from "./Whiteboard.module.css";
import { useOthers } from "../../liveblocks.config";
import SomeoneIsTyping from "../SomeoneIsTyping/SomeoneIsTyping";
import WhoIsHere from "../WhoIsHere/WhoIsHere";
import { TeamList } from "../TeamList/TeamList";
import Dialog from "../../components/Dialog/Dialog";
import axios from "axios";
import { Canvas } from "../Canvas/Canvas";
import { ClientSideSuspense } from "@liveblocks/react";
import { useSession } from "next-auth/react";
import { Spinner } from "../../primitives/Spinner/Spinner";
import { useState } from "react";

export function Whiteboard() {
  const { data: session } = useSession();
  const [userInput, setUserInput] = useState("");

  const loading = (
    <div className={styles.loading}>
      <Spinner size={24} />
    </div>
  );

  const [output, setOutput] = useState("");

  async function handleUserInput(userInput: string) {
    try {
      const response = await axios.post("http://localhost:8000/process", {
        user_input: userInput,
      });
      setOutput(response.data.response);
    } catch (error) {
      console.error("Error processing user input:", error);
    }
  }

  return (
    <ClientSideSuspense fallback={loading}>
      {() => (
        <>
          <Canvas
            currentUser={session?.user.info ?? null}
            handleUserInput={handleUserInput}
          />
          <Dialog output={output} />
        </>
      )}
    </ClientSideSuspense>
  );
}
