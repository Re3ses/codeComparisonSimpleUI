import React from "react";
import { Modal, ModalContent, ModalHeader, ModalBody, ModalFooter, useDisclosure } from "@nextui-org/modal";
import { Button } from "@nextui-org/react";

interface queryOptionsProprs {
    onTokenizerChange: (event: any) => void;
    tokenizer: string;
}

const QueryOptions: React.FC<queryOptionsProprs> = ({ onTokenizerChange, tokenizer }) => {
    const { isOpen, onOpen, onClose } = useDisclosure();

    const handleQuerySettings = (event: any) => {
        event.preventDefault(); // Prevent form submission
        onClose(); // Close the modal after saving settings
    };

    return (
        <div>
            <Button className="px-3 py-2 bg-yellow-100/30" onClick={onOpen}>
                Options
            </Button>
            <Modal size="xs" isOpen={isOpen} onClose={onClose} className="bg-gray-900">
                <ModalContent>
                    <ModalHeader className="flex flex-col gap-1">Query Settings</ModalHeader>
                    <ModalBody>
                        <form onSubmit={handleQuerySettings}>
                            <div className="flex flex-col space-y-2">
                                <label className="font-medium">Tokenizer Type:</label>
                                <div className="flex flex-col space-y-1">
                                    <label className="flex items-center">
                                        <input
                                            type="radio"
                                            name="tokenizer"
                                            value="char"
                                            checked={tokenizer === "char"}
                                            onChange={onTokenizerChange}
                                            className="mr-2"
                                        />
                                        Char
                                    </label>
                                    <label className="flex items-center">
                                        <input
                                            type="radio"
                                            name="tokenizer"
                                            value="subword"
                                            checked={tokenizer === "subword"}
                                            onChange={onTokenizerChange}
                                            className="mr-2"
                                        />
                                        Subword
                                    </label>
                                    <label className="flex items-center">
                                        <input
                                            type="radio"
                                            name="tokenizer"
                                            value="word"
                                            checked={tokenizer === "word"}
                                            onChange={onTokenizerChange}
                                            className="mr-2"
                                        />
                                        Word
                                    </label>
                                </div>
                            </div>
                        </form>
                    </ModalBody>
                    <ModalFooter>
                        <Button
                            type="submit"
                            className="px-3 py-2 bg-blue-500 hover:bg-blue-700 mt-4"
                            onPress={onClose}
                        >
                            Save Settings
                        </Button>
                    </ModalFooter>
                </ModalContent>
            </Modal>
        </div>
    );
}

export default QueryOptions;