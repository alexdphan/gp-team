import { ComponentProps, forwardRef, ReactNode } from "react";
import clsx from "clsx";
import styles from "./Document.module.css";
import { WhiteboardSidebar } from "../../components/Whiteboard/WhiteboardSidebar";

interface Props extends ComponentProps<"div"> {
  header: ReactNode;
}

export const DocumentLayout = forwardRef<HTMLElement, Props>(
  ({ children, header, className, ...props }, ref) => {
    return (
      <div className={clsx(className, styles.container)} {...props}>
        <header className={styles.header}>{header}</header>
        
        {/* <WhiteboardSidebar className="absolute top-0 right-0 h-full w-1/4 bg-white border-l border-gray-300 p-4" isSidebarOpen={false} setIsSidebarOpen={function (): void {
          throw new Error("Function not implemented.");
        } } /> */}
        <main className={styles.main} ref={ref}>
          {children}
        </main>
      </div>
    );
  }
);
