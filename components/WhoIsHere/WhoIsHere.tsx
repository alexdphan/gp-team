import { ClientSideSuspense } from "@liveblocks/react";
import { RoomProvider, useOthers } from "../../liveblocks.config";

function WhoIsHere() {
  const userCount = useOthers((others) => others.length);

  return (
    <div className="inline-flex items-center px-2 py-1 bg-[#f8e4cb] rounded-md font-medium text-sm m-3">
      <span className="text-gray-700">{userCount}</span>
      <span className="text-gray-600 ml-1">
        {userCount === 1 ? "user online" : "users online"}
      </span>
    </div>
  );
}

export default WhoIsHere;

// text-black px-5 mx-5 py-2 my-2 who_is_here font-bold bg-orange-200 w-40 border border-black rounded-lg
