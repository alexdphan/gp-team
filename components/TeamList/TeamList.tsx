import * as React from "react";

import { ScrollArea } from "../ui/scroll-area";
import { Separator } from "../ui/separator";
import { Avatar, AvatarFallback, AvatarImage } from "../ui/avatar";

// Example data for team members
const teamMembers = Array.from({ length: 5 }).map((_, i) => ({
  name: `Member ${i + 1}`,
  avatarUrl: "https://github.com/shadcn.png", // Replace with real avatar URLs
  description: `Short description for Member ${i + 1}`,
}));

export function TeamList() {
  return (
    <ScrollArea className=" rounded-md border">
      <div className="p-4">
        <h4 className="mb-4 text-sm font-medium leading-none">Team Members</h4>
        {teamMembers.map((member) => (
          <React.Fragment key={member.name}>
            <div className="flex items-center space-x-3 text-sm">
              <Avatar>
                <AvatarImage
                  src={member.avatarUrl}
                  alt={member.name}
                  sizes="sm"
                />
              </Avatar>
              <div>
                <div>{member.name}</div>
                <div className="text-xs text-gray-600">
                  {member.description}
                </div>
              </div>
            </div>
            <Separator className="my-2" />
          </React.Fragment>
        ))}
      </div>
    </ScrollArea>
  );
}
