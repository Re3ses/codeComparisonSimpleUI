import React from "react";

interface Comparison {
    filename: string;
    structural_similarity: number;
    token_similarity: number;
    tfidf_similarity: number;
    semantic_similarity: number;
    combined_similarity: number;
    potential_plagiarism: boolean;
}

interface ComparisonResult {
    file: string;
    comparisons: Comparison[];
}

interface ComparisonResultsProps {
    comparisonResult: ComparisonResult[];
}

const ComparisonResults: React.FC<ComparisonResultsProps> = ({ comparisonResult }) => (
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
                            <th className="p-2 border text-right">Structural Similarity</th>
                            <th className="p-2 border text-right">Token Similarity</th>
                            <th className="p-2 border text-right">TF-IDF Similarity</th>
                            <th className="p-2 border text-right">Semantic Similarity</th>
                            <th className="p-2 border text-right">Combined Similarity</th>
                            <th className="p-2 border text-right">Potential Plagiarism</th>
                        </tr>
                    </thead>
                    <tbody>
                        {result.comparisons.map((comparison, i) => (
                            <tr
                                key={i}
                                className={comparison.potential_plagiarism ? "bg-red-100" : ""}
                            >
                                <td className="p-2 border">{comparison.filename}</td>
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
                                <td className={`p-2 border text-right ${
                                    comparison.potential_plagiarism ? "bg-red-500" : "bg-green-600"
                                }`}>
                                    {comparison.potential_plagiarism ? "Yes" : "No"}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        ))}
    </div>
);

export default ComparisonResults;