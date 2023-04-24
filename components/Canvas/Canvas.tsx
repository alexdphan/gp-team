import { ChangeEvent, FocusEvent, PointerEvent, useRef, useState } from "react";
import clsx from "clsx";
import { LiveObject } from "@liveblocks/client";
import { nanoid } from "nanoid";
import {
  useCanRedo,
  useCanUndo,
  useHistory,
  useMutation,
  useSelf,
  useStorage,
  UserMeta,
  useUpdateMyPresence,
} from "../../liveblocks.config";
import styles from "../Whiteboard/Whiteboard.module.css";
import { WhiteboardNote } from "../Whiteboard/WhiteboardNote";
import { Cursors } from "../Cursors";
import { useBoundingClientRectRef } from "../../utils";
import { TeamList } from "../TeamList/TeamList";
import WhoIsHere from "../WhoIsHere/WhoIsHere";
import SomeoneIsTyping from "../SomeoneIsTyping/SomeoneIsTyping";

type HandleUserInput = (userInput: string) => Promise<void>;

interface Props extends React.ComponentProps<"div"> {
  currentUser: UserMeta["info"] | null;
  handleUserInput: HandleUserInput;
}

export function Canvas({
  currentUser,
  handleUserInput,
  className,
  style,
  ...props
}: Props) {
  // Canvas component code here

  const updateMyPresence = useUpdateMyPresence();
  const [draft, setDraft] = useState("");
  const elementRef = useRef<HTMLDivElement | null>(null); // Create a ref for the element

  return (
    <div
      className={clsx(styles.whiteboard, className)}
      style={style}
      {...props}
    >
      {/* Add the JSX for the Canvas component */}
      <div className="absolute top-5 right-0 mt-[headerHeight] mr-4 md:block hidden">
        <TeamList />
      </div>
      <Cursors element={elementRef} />
      <div
        ref={elementRef} // Set the ref for the element
        className="flex flex-col items-center"
      >
        <div className="flex flex-row space-x-1">
          <WhoIsHere />
          <SomeoneIsTyping />
        </div>
        <div className={styles.toolbar}>
          <input
            type="text"
            placeholder="Type something"
            value={draft}
            onChange={(e) => {
              setDraft(e.target.value);
              updateMyPresence({ isTyping: true });
            }}
            onKeyDown={async (e) => {
              if (e.key === "Enter") {
                updateMyPresence({ isTyping: false });
                await handleUserInput(draft);
                setDraft("");
              }
            }}
            onBlur={() => updateMyPresence({ isTyping: false })}
            className="w-full" // Add Tailwind CSS class to set the width to 100%
          />
        </div>
      </div>
    </div>
  );
}
