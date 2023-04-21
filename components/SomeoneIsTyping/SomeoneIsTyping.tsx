import { ClientSideSuspense } from "@liveblocks/react";
import {
  RoomProvider,
  useOthers,
  useUpdateMyPresence,
} from "../../liveblocks.config";
import { useState } from "react";
import WhoIsHere from "../WhoIsHere/WhoIsHere";

/* WhoIsHere */

function SomeoneIsTyping() {
  const someoneIsTyping = useOthers((others) =>
    others.some((other) => other.presence.isTyping)
  );

  return someoneIsTyping ? (
    <div className="inline-flex items-center px-2 py-1 bg-[#E0E0E0] rounded-md font-medium text-sm m-3">
      <div>Someone is typing...</div>
    </div>
  ) : null;
}

export default SomeoneIsTyping;
