<?php

declare(strict_types=1);

namespace App\AI;

use Psr\Http\Message\ResponseInterface;

trait StreamingTrait
{
    private array $data = [];

    protected function extractData(ResponseInterface $response, bool $stream, callable $streamFunction): array
    {
        if ($stream) {
            return $this->extractFromStream($response, $streamFunction);
        }
        $responseData = json_decode($response->getBody()->getContents(), true);

        return $this->extractFromJson($responseData);
    }

    protected function extractFromJson(array $responseData): array
    {
        return [
            'content' => $responseData['choices'][0]['message']['content'] ?? '',
            'output_tokens' => $responseData['usage']['completion_tokens'] ?? 0,
            'input_tokens' => $responseData['usage']['prompt_tokens'] ?? 0,
        ];
    }

    protected function extractFromStream($response, callable $streamFunction): array
    {
        $stream = $response->getBody();

        $this->data = [
            'content' => '',
            'input_tokens' => 0,
            'output_tokens' => 0,
        ];

        foreach ($this->readStream($stream) as $event) {
            $this->extractTokens($event, $streamFunction);
        }

        return $this->data;
    }

    protected function extractTokens(array $event, callable $streamFunction)
    {
        if (isset($event['choices'][0]['delta']['content'])) {
            $this->data['content'] .= $event['choices'][0]['delta']['content'];
            $streamFunction($event['choices'][0]['delta']['content']);
        }
        $usage = $event['usage'] ?? $event['x_groq']['usage'] ?? [];
        if (isset($usage['prompt_tokens'])) {
            $this->data['input_tokens'] = $usage['prompt_tokens'];
        }
        if (isset($usage['completion_tokens'])) {
            $this->data['output_tokens'] = $usage['completion_tokens'];
        }
    }

    protected function readStream($stream)
    {
        $buffer = '';
        while (! $stream->eof()) {
            $buffer .= $stream->read(1024);
            while (($pos = strpos($buffer, "\n")) !== false) {
                $line = substr($buffer, 0, $pos);
                $buffer = substr($buffer, $pos + 1);

                if (str_starts_with($line, 'data: ')) {
                    $data = json_decode(trim(substr($line, 5)), true);
                    if ($data) {
                        yield $data;
                    }
                } elseif (str_ends_with($line, '}')) {
                    $data = json_decode(trim($line), true);
                    if ($data) {
                        yield $data;
                    }
                }
            }
        }
    }
}
