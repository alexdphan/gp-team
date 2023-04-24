interface DialogProps {
  output: string; // Define the output prop as a string
}

const Dialog: React.FC<DialogProps> = ({ output }) => {
  return (
    <div>
      {/* Render the output prop directly */}
      {output}
    </div>
  );
};

export default Dialog;
