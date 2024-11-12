import React, { useState } from "react";
import {Modal, ModalContent, ModalHeader, ModalBody, ModalFooter, useDisclosure} from "@nextui-org/modal";
import {Button} from "@nextui-org/react";

const CodeComparisonApp = () => {
  const [files, setFiles] = useState([]);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [tokenizer, setTokenizer] = useState("char");

  // modal
  const { isOpen, onOpen, onClose } = useDisclosure();

  const handleQuerySettings = (event: any) => {
    event.preventDefault(); // Prevent form submission
    onClose(); // Close the modal after saving settings
  };
  
  const handleTokenizerChange = (event: any) => {
    setTokenizer(event.target.value);
    console.log(event.target.value);
  };


  const handleFileUpload = (event: any) => {
    const newFiles = Array.from(event.target.files).map((file) => ({
      name: file.name,
      content: URL.createObjectURL(file),
      language: "python",
    }));
    setFiles([...files, ...newFiles]);
  };

  const handleCompare = async () => {
    setLoading(true)
    try {
      console.log("Comparing code");
      const fileData = {};
      files.forEach((file) => {
        fileData[file.name] = {
          code: file.content,
          language: file.language,
        };
      });

      const response = await fetch("http://127.0.0.1:5000/compare", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ 
          submissions: fileData,
          tokenizer: tokenizer
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setComparisonResult(data);
      } else {
        console.error("Error comparing code:", await response.json());
      }
    } catch (error) {
      console.error("Error comparing code:", error);
    } finally {
      setLoading(false)
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-w-full">
      <h1 className="text-2xl font-bold">Code Comparison App</h1>
      <div className="max-w-4xl p-6 space-y-4 flex flex-col min-w-[70%]">
        <div className="">
          <label htmlFor="file-input" className="block font-medium mb-2">
            Upload Files
          </label>
          <div className="flex flex-row gap-8 items-center">
            <input
              id="file-input"
              type="file"
              multiple
              accept=".py"
              className="block w-full border rounded-md p-2"
              onChange={handleFileUpload}
            />
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
                              onChange={handleTokenizerChange}
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
                              onChange={handleTokenizerChange}
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
                              onChange={handleTokenizerChange}
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

              {loading ? <p>loading</p> :
              <button
                onClick={handleCompare}
                className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
              >
                Compare Code
              </button>}
            </div>
          </div>
        </div>

        <div className="flex flex-row gap-3">
          <div className="flex flex-row w-1/6">
            <div>
              <h2 className="font-medium mb-2">Uploaded Files</h2>
              <ul className="list-disc pl-6">
                {files.map((file, index) => (
                  <li key={index}>{file.name}</li>
                ))}
              </ul>
            </div>
          </div>

          <div className="">
            {comparisonResult && (
              <div>
                <h2 className="font-medium mb-2">Comparison Results:</h2>
                {comparisonResult.map((result, index) => (
                  <div key={index} className="mb-4">
                    <h3 className="font-medium">
                      Compared File: {result.file}
                    </h3>
                    <table className="w-full border-collapse">
                      <thead>
                        <tr>
                          <th className="p-2 border text-left">Filename</th>
                          <th className="p-2 border text-right">
                            Structural Similarity
                          </th>
                          <th className="p-2 border text-right">
                            Token Similarity
                          </th>
                          <th className="p-2 border text-right">
                            TF-IDF Similarity
                          </th>
                          <th className="p-2 border text-right">
                            Semantic Similarity
                          </th>
                          <th className="p-2 border text-right">
                            Combined Similarity
                          </th>
                          <th className="p-2 border text-right">
                            Potential Plagiarism
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.comparisons.map((comparison, i) => (
                          <tr
                            key={i}
                            className={
                              comparison.potential_plagiarism ? "bg-red-100" : ""
                            }
                          >
                            <td className="p-2 border">
                              {comparison.filename}
                            </td>
                            <td className="p-2 border text-right">
                              {comparison.structural_similarity.toFixed(2)}%
                            </td>
                            <td className="p-2 border text-right">
                              {comparison.token_similarity.toFixed(2)}%
                            </td>
                            <td className="p-2 border text-right">
                              {comparison.tfidf_similarity.toFixed(2)}%
                            </td>
                            <td className="p-2 border text-right">
                              {comparison.semantic_similarity.toFixed(2)}%
                            </td>
                            <td className="p-2 border text-right">
                              {comparison.combined_similarity.toFixed(2)}%
                            </td>
                            <td className={`p-2 border text-right ${comparison.potential_plagiarism ? "bg-red-500" : "bg-green-600"}`}>
                              {comparison.potential_plagiarism ? "Yes" : "No"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
  );
};

export default CodeComparisonApp;