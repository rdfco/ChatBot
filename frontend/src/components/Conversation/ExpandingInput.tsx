import {
  PaperAirplaneIcon,
  Cog6ToothIcon,
  ShieldExclamationIcon,
  ShieldCheckIcon,
} from "@heroicons/react/24/outline";
import { SetStateAction, forwardRef, useRef, useState } from "react";

import { Switch, SwitchField, SwitchGroup } from "../Catalyst/switch";
import { Description, Fieldset, Label } from "../Catalyst/fieldset";

import { Transition } from "@headlessui/react";

import {
  useGetRelatedConnection,
  getMessageOptions,
  usePatchMessageOptions,
} from "@/hooks";
import { useClickOutside } from "../Library/utils";
import { useQuery } from "@tanstack/react-query";

type ExpandingInputProps = {
  onSubmit: (value: string) => void;
  disabled: boolean;
};

function classNames(...classes: string[]) {
  return classes.filter(Boolean).join(" ");
}

type MessageSettingsPopupProps = {
  isShown: boolean;
  setIsShown: (val: boolean) => void;
  settingsCogRef: React.MutableRefObject<HTMLDivElement | null>;
};

const MessageSettingsPopup: React.FC<MessageSettingsPopupProps> = ({
  isShown,
  setIsShown,
  settingsCogRef: cogRef,
}) => {
  const currConnection = useGetRelatedConnection();
  const { data: messageOptions } = useQuery(
    getMessageOptions(currConnection?.id)
  );
  const { mutate: patchMessageOptions } = usePatchMessageOptions(
    currConnection?.id
  );
  const settingsPopupRef = useRef<HTMLDivElement | null>(null);
  useClickOutside([settingsPopupRef, cogRef], () => {
    setIsShown(false);
  });

  return (
    <Transition
      show={isShown}
      enter="transition-opacity duration-200"
      enterFrom="opacity-0"
      enterTo="opacity-100"
      leave="transition-opacity duration-200"
      leaveFrom="opacity-100"
      leaveTo="opacity-0"
    >
      <div
        ref={settingsPopupRef}
        className={classNames(
          "absolute left-0 bottom-1 border p-4 bg-gray-900 border-gray-600 rounded-xl"
        )}
      >
        <Fieldset>
          <SwitchGroup>
            <SwitchField>
              <Label className="flex items-center">Data Security </Label>
              <Description>Hide your data from the AI model</Description>
              <Switch
                color="green"
                checked={messageOptions?.secure_data}
                onChange={(checked) =>
                  patchMessageOptions({ secure_data: checked })
                }
                name="data_security"
              />
            </SwitchField>
          </SwitchGroup>
        </Fieldset>
      </div>
    </Transition>
  );
};

const ExpandingInput = forwardRef<HTMLTextAreaElement, ExpandingInputProps>(
  ({ onSubmit, disabled }, ref) => {
    const [inputValue, setInputValue] = useState("");
    const [messageSettingsShown, setMessageSettingsShown] = useState(false);
    const currConnection = useGetRelatedConnection();
    const { data: messageOptions } = useQuery(
      getMessageOptions(currConnection?.id)
    );
    const settingsCogRef = useRef<HTMLDivElement | null>(null);

    const handleChange = (e: {
      target: {
        value: SetStateAction<string>;
        style: { height: string };
        scrollHeight: any;
      };
    }) => {
      setInputValue(e.target.value);
      e.target.style.height = "auto"; // Reset textarea height
      e.target.style.height = `${e.target.scrollHeight}px`; // Set textarea height based on content
    };

    const handleSubmit = () => {
      if (disabled) return;
      if (inputValue.length === 0) return;
      onSubmit(inputValue);
      setInputValue("");
    };

    const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();

        // Reset textarea height
        e.currentTarget.style.height = "auto";
      }
    };

    return (
      <div className="flex flex-col justify-center w-full relative mb-4">
        <textarea
          name="email"
          id="email"
          className={classNames(
            disabled
              ? "placeholder:text-gray-600 text-gray-800 dark:text-gray-400 dark:bg-gray-800 focus:ring-0"
              : "placeholder:text-gray-400 text-gray-900 dark:text-gray-200 dark:bg-gray-900",
            "block rounded-xl border p-4 shadow-sm sm:text-md sm:leading-6 resize-none dark:border-gray-600 pr-12 overflow-y-hidden mr-1"
          )}
          style={{ height: "auto" }}
          rows={1}
          placeholder="Enter your message here..."
          value={inputValue}
          onChange={handleChange}
          onKeyDown={handleKeyPress}
          ref={ref}
        />
        <div className="absolute left-0 top-0 w-full">
          <MessageSettingsPopup
            isShown={messageSettingsShown}
            setIsShown={setMessageSettingsShown}
            settingsCogRef={settingsCogRef}
          />
        </div>
        <div
          onClick={handleSubmit}
          className={classNames(
            inputValue.length > 0 && !disabled
              ? "dark:text-gray-700 dark:bg-gray-300 dark:hover:cursor-pointer"
              : "",
            "group absolute right-0 mr-4 -rotate-90 dark:text-gray-400 p-1 rounded-md transition-all duration-150"
          )}
        >
          <PaperAirplaneIcon
            className={classNames(
              inputValue.length > 0 ? "group-hover:-rotate-6" : "",
              "h-6 w-6 [&>path]:stroke-[2]"
            )}
          ></PaperAirplaneIcon>
        </div>
      </div>
    );
  }
);

export default ExpandingInput;
